"""
CT 预处理核心模块
====================

对标 VoxRep 的预处理流水线, 并针对临床 CT 做了以下扩展:

VoxRep 原始流程:
  100×100×16 体素网格 → Z轴切16层 → 各层 pad 到 112 → resize 到 224
  → 4×4 拼成 896×896 → 送入 Gemma3 编码器

本模块适配临床 CT 的改进:
  1. NIfTI 读取 + LPS 方向标准化 (临床 CT 方向不固定)
  2. HU 值清洗 (NaN/Inf/越界)
  3. 自适应裁剪 (去除扫描床/空气层, VoxRep 不需要因为是合成数据)
  4. 智能切片采样: uniform / weighted / density 三种策略
     - VoxRep 的 16 层是固定的, 临床 CT 有几百层, 需要采样
  5. 单切片 resize 到 224×224, 与 VLM 的 14×14 patch 对齐
     - VoxRep 也是 resize 到 224, 这里保持一致
  6. 多窗位变换 (VoxRep 直接用 RGB 颜色, CT 需要窗宽窗位)
  7. 拼成 grid_rows × grid_cols 的大图 (VoxRep 用 4×4, 我们用 8×8)

输出:
  tensors_dict: {window_name: np.ndarray of shape (N, H, W, 3) uint8}
  metadata:     PreprocessMetadata (包含层间距等关键信息, 注入 prompt)
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
import nibabel.orientations as ornt
from scipy.ndimage import zoom

from configs.config import cfg, WindowConfig

logger = logging.getLogger(__name__)


# ================================================================== #
# 预处理元数据
# ================================================================== #
@dataclass
class PreprocessMetadata:
    """
    预处理过程的全量元数据

    用途:
      1. 注入 prompt, 让 VLM 理解空间关系 (层间距、层号、扫描范围)
      2. 溯源: 从报告追溯到原始数据的任何位置
      3. 评测: 记录预处理参数便于消融实验
    """
    file_path: str
    original_shape: Tuple[int, int, int]           # (Z, Y, X) after orientation
    original_spacing_mm: Tuple[float, float, float] # (dz, dy, dx) in mm
    normalized_shape: Tuple[int, int, int]         # XY 间距归一化后的 shape
    normalized_spacing_mm: Tuple[float, float, float]  # 归一化后的 spacing
    cropped_z_range: Tuple[int, int]               # 裁剪后保留的 Z 范围 [start, end]
    cropped_depth: int                             # 裁剪后的总层数
    sampled_z_indices: List[int]                   # 采样的层号 (基于裁剪后)
    sampled_z_indices_original: List[int]          # 采样的层号 (映射回原始卷)
    effective_slice_spacing_mm: float              # 采样后的有效层间距
    target_slices: int
    sampling_strategy: str
    windows_used: List[str]
    slice_size: int
    tiled_image_size: Tuple[int, int]              # 最终拼图尺寸 (H, W)
    scan_coverage_mm: float                        # 扫描覆盖的 Z 轴距离 (mm)
    xy_spacing_normalized: bool                    # 是否做了 XY 间距归一化

    def to_dict(self) -> dict:
        return asdict(self)

    def to_prompt_context(self) -> str:
        """生成可直接注入 prompt 的空间上下文描述"""
        dz, dy, dx = self.original_spacing_mm
        nz, ny, nx = self.normalized_spacing_mm
        norm_note = (
            f", 已归一化到 {ny:.2f}×{nx:.2f}mm 面内分辨率"
            if self.xy_spacing_normalized else ""
        )
        return (
            f"扫描参数: 原始矩阵 {self.original_shape[1]}×{self.original_shape[2]}, "
            f"共 {self.original_shape[0]} 层, "
            f"原始层间距 {dz:.2f}mm, "
            f"原始面内分辨率 {dy:.2f}×{dx:.2f}mm{norm_note}。"
            f"有效裁剪后 {self.cropped_depth} 层, "
            f"均匀采样 {self.target_slices} 层 (有效层间距 ~{self.effective_slice_spacing_mm:.1f}mm), "
            f"Z轴覆盖 ~{self.scan_coverage_mm:.0f}mm。"
        )


# ================================================================== #
# CT 预处理器
# ================================================================== #
class CTProcessor:
    """CT 数据预处理流水线"""

    def __init__(self):
        self.target_slices = cfg.target_slices

    # ---------------------------------------------------------------- #
    # 1. 方向标准化
    # ---------------------------------------------------------------- #
    def standardize_orientation(
        self, ct_img: nib.Nifti1Image
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        强制转化为 LPS 坐标系并转置为 (Z, Y, X)

        放射学惯例:
          - Z 轴: 自下 (足) 到上 (头), 即 S 方向
          - Y 轴: 由前 (腹) 到后 (背), 即 P 方向
          - X 轴: 由右到左, 即 L 方向

        返回:
          volume:  (Z, Y, X) float32 数组, HU 值
          spacing: (dz, dy, dx) 体素间距, 单位 mm
        """
        orig_ornt = nib.io_orientation(ct_img.affine)
        targ_ornt = ornt.axcodes2ornt(("L", "P", "S"))
        transform = ornt.ornt_transform(orig_ornt, targ_ornt)
        aligned = ornt.apply_orientation(ct_img.get_fdata(), transform)

        # 获取变换后的体素间距
        # header.get_zooms() 对应原始轴序, 需要按 transform 重映射
        orig_zooms = np.array(ct_img.header.get_zooms()[:3], dtype=np.float64)
        # transform[:, 0] 记录了原始轴到目标轴的映射
        axis_mapping = transform[:, 0].astype(int)
        reordered_zooms = orig_zooms[axis_mapping]

        # aligned 的轴顺序是 (L, P, S), 转置为 (S, P, L) = (Z, Y, X)
        volume = np.transpose(aligned, (2, 1, 0)).astype(np.float32)
        # 对应 spacing 也要转置: (S_zoom, P_zoom, L_zoom) = (dz, dy, dx)
        spacing = (
            float(reordered_zooms[2]),   # S -> dz
            float(reordered_zooms[1]),   # P -> dy
            float(reordered_zooms[0]),   # L -> dx
        )
        return volume, spacing

    # ---------------------------------------------------------------- #
    # 2. 数据清洗
    # ---------------------------------------------------------------- #
    def sanitize_hu(self, volume: np.ndarray) -> np.ndarray:
        """
        处理 NaN/Inf 并限制 HU 值到物理合理区间

        临床 CT 的 HU 范围:
          - 空气: -1024
          - 水: 0
          - 软组织: 20~80
          - 骨骼: 400~1000+
          - 金属植入物: 可达 3071
        """
        if not np.all(np.isfinite(volume)):
            n_bad = int(np.sum(~np.isfinite(volume)))
            logger.warning(f"检测到 {n_bad} 个 NaN/Inf 体素, 替换为 {cfg.hu_min}")
            volume = np.nan_to_num(
                volume, nan=cfg.hu_min, posinf=cfg.hu_max, neginf=cfg.hu_min
            )
        return np.clip(volume, cfg.hu_min, cfg.hu_max)

    # ---------------------------------------------------------------- #
    # 2.5 XY 面内间距归一化 (借鉴 CT-CLIP)
    # ---------------------------------------------------------------- #
    def normalize_xy_spacing(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        将 volume 的 XY 面内间距归一化到 cfg.target_xy_spacing_mm

        借鉴自 CT-CLIP: 所有 CT 先 resample 到统一物理分辨率再送入模型
        我们只做 XY (面内), 不做 Z (交给切片采样处理), 避免 3D 插值伪影

        流程:
          1. 按物理间距比例 resample XY 平面 (双线性)
          2. center-crop 或 pad 到 cfg.target_xy_size

        输入:
          volume:  (Z, Y, X) float32 HU
          spacing: (dz, dy, dx) mm
        输出:
          new_volume:  (Z, target_xy_size, target_xy_size) float32 HU
          new_spacing: (dz, target_xy_spacing_mm, target_xy_spacing_mm)
        """
        if not cfg.normalize_xy_spacing:
            return volume, spacing

        dz, dy, dx = spacing
        target_sp = cfg.target_xy_spacing_mm
        target_size = cfg.target_xy_size

        # 1. XY 重采样: 计算缩放因子
        # 新尺寸 = 原尺寸 * (原间距 / 目标间距)
        zoom_y = dy / target_sp
        zoom_x = dx / target_sp

        # 只对 XY 做缩放, Z 轴保持不变
        # order=1 双线性, 对 HU 连续值合适
        resampled = zoom(
            volume,
            zoom=(1.0, zoom_y, zoom_x),
            order=1,
            mode="nearest",
        )
        logger.info(
            f"XY 重采样: spacing ({dy:.2f}, {dx:.2f}) -> ({target_sp}, {target_sp}) mm, "
            f"shape {volume.shape[1:]} -> {resampled.shape[1:]}"
        )

        # 2. Center-crop 或 pad 到 target_size × target_size
        z, h, w = resampled.shape
        new_volume = np.full(
            (z, target_size, target_size),
            fill_value=float(cfg.hu_min),  # 用空气 HU 填充, 符合 CT 物理意义
            dtype=np.float32,
        )

        # 计算 Y 轴的 crop / pad 偏移
        if h >= target_size:
            # 需要 crop: 取中央 target_size
            src_y_start = (h - target_size) // 2
            src_y_end = src_y_start + target_size
            dst_y_start, dst_y_end = 0, target_size
        else:
            # 需要 pad: 居中放置
            src_y_start, src_y_end = 0, h
            dst_y_start = (target_size - h) // 2
            dst_y_end = dst_y_start + h

        # 同上处理 X 轴
        if w >= target_size:
            src_x_start = (w - target_size) // 2
            src_x_end = src_x_start + target_size
            dst_x_start, dst_x_end = 0, target_size
        else:
            src_x_start, src_x_end = 0, w
            dst_x_start = (target_size - w) // 2
            dst_x_end = dst_x_start + w

        new_volume[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            resampled[:, src_y_start:src_y_end, src_x_start:src_x_end]

        new_spacing = (dz, target_sp, target_sp)
        logger.info(
            f"XY 中心裁剪/填充: -> ({target_size}, {target_size}), "
            f"物理覆盖 {target_size * target_sp:.0f}mm × {target_size * target_sp:.0f}mm"
        )
        return new_volume, new_spacing

    # ---------------------------------------------------------------- #
    # 3. 自适应裁剪
    # ---------------------------------------------------------------- #
    def get_valid_anatomical_region(
        self, volume: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        基于强度统计裁掉头尾纯空气层和扫描床外区域

        策略:
          - 计算每层的第 crop_intensity_percentile 分位数
          - 高于动态阈值 (min + range * ratio) 的层被认为包含人体组织
          - 在有效范围两端各额外保留 crop_margin_slices 层作为缓冲
        """
        depth = volume.shape[0]
        global_min = float(np.min(volume))
        global_max = float(np.max(volume))
        dyn_threshold = global_min + (global_max - global_min) * cfg.crop_threshold_ratio

        valid_slices = []
        for z in range(depth):
            percentile_val = np.percentile(volume[z], cfg.crop_intensity_percentile)
            if percentile_val > dyn_threshold:
                valid_slices.append(z)

        if not valid_slices:
            logger.warning("自适应裁剪未找到有效层, 保留全部")
            return volume, (0, depth - 1)

        # 加入安全边距
        start_z = max(0, valid_slices[0] - cfg.crop_margin_slices)
        end_z = min(depth - 1, valid_slices[-1] + cfg.crop_margin_slices)

        logger.info(
            f"自适应裁剪 Z: [{start_z}:{end_z}] "
            f"(有效 {len(valid_slices)} 层, 原始 {depth} 层, "
            f"保留 {end_z - start_z + 1} 层)"
        )
        return volume[start_z:end_z + 1], (start_z, end_z)

    # ---------------------------------------------------------------- #
    # 4. 切片采样
    # ---------------------------------------------------------------- #
    def sample_slice_indices(
        self, depth: int, volume: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        根据采样策略返回需要提取的层 index 列表

        策略:
          - uniform: 均匀采样, 首尾必含 (类似 VoxRep, 其 16 层全取)
          - weighted: 中段加权 (20%-80% 区域分配 70% 的采样点)
          - density: 基于信息密度的自适应采样 (需要 volume)
                     高信息量的层分配更多采样点

        当 depth <= target_slices 时全部保留, 末尾重复补齐
        """
        target = self.target_slices

        # Case: 切片数不足, 纯黑值补齐
        if depth <= target:
            indices = list(range(depth))
            # while len(indices) < target:
            #     indices.append(depth - 1)
            return indices

        strategy = cfg.sampling_strategy

        if strategy == "uniform":
            return [int(round(i)) for i in np.linspace(0, depth - 1, target)]

        if strategy == "weighted":
            n_top = max(1, int(target * 0.15))
            n_bot = max(1, int(target * 0.15))
            n_mid = target - n_top - n_bot
            z1 = int(depth * 0.2)
            z2 = int(depth * 0.8)

            idx_top = np.linspace(0, max(z1 - 1, 0), n_top, dtype=int).tolist()
            idx_mid = np.linspace(z1, max(z2 - 1, z1), n_mid, dtype=int).tolist()
            idx_bot = np.linspace(z2, depth - 1, n_bot, dtype=int).tolist()

            indices = sorted(set(idx_top + idx_mid + idx_bot))
            # 去重后不够时用均匀补齐
            if len(indices) < target:
                fill = np.linspace(0, depth - 1, target, dtype=int).tolist()
                for v in fill:
                    if len(indices) >= target:
                        break
                    if v not in indices:
                        indices.append(v)
                indices = sorted(indices)[:target]
            return indices

        if strategy == "density":
            if volume is None:
                logger.warning("density 采样需要 volume, 回退到 uniform")
                return [int(round(i)) for i in np.linspace(0, depth - 1, target)]

            # 计算每层的信息密度: 使用标准差 + 非空气体素占比
            scores = np.zeros(depth, dtype=np.float64)
            for z in range(depth):
                layer = volume[z]
                std_val = float(np.std(layer))
                # 非空气比例 (HU > -800 的体素)
                tissue_ratio = float(np.mean(layer > -800))
                scores[z] = std_val * (0.3 + 0.7 * tissue_ratio)

            # 首尾层强制选中
            scores[0] = max(scores[0], np.max(scores))
            scores[-1] = max(scores[-1], np.max(scores))

            # 平滑 scores 避免采样过于集中
            kernel_size = max(3, depth // 20)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = np.ones(kernel_size) / kernel_size
            scores_smooth = np.convolve(scores, kernel, mode="same")

            # 基于 CDF 采样
            cdf = np.cumsum(scores_smooth)
            cdf = cdf / cdf[-1]
            sample_points = np.linspace(0, 1, target)
            indices = []
            for sp in sample_points:
                idx = int(np.searchsorted(cdf, sp))
                idx = min(idx, depth - 1)
                indices.append(idx)

            # 去重 + 确保首尾
            indices = sorted(set(indices))
            if 0 not in indices:
                indices = [0] + indices
            if depth - 1 not in indices:
                indices.append(depth - 1)

            # 补齐到 target
            if len(indices) < target:
                fill = np.linspace(0, depth - 1, target, dtype=int).tolist()
                for v in fill:
                    if len(indices) >= target:
                        break
                    if v not in indices:
                        indices.append(v)
                indices = sorted(indices)
            return indices[:target]

        raise ValueError(f"未知采样策略: {strategy}")

    # ---------------------------------------------------------------- #
    # 5. 单切片 resize
    # ---------------------------------------------------------------- #
    def resize_slice(self, slice_2d: np.ndarray, target_size: int) -> np.ndarray:
        """
        对单张 (H, W) 切片进行双线性插值 resize 到 (target_size, target_size)

        VoxRep 的做法: 100×100 → pad 到 112×112 → resize 到 224×224 (双线性)
        我们简化为直接 resize (临床 CT 通常是 512×512, 不需要 pad)

        对于非正方形的输入, 先 pad 到正方形再 resize, 避免纵横比失真
        """
        h, w = slice_2d.shape
        if (h, w) == (target_size, target_size):
            return slice_2d

        # 如果不是正方形, 先 pad 到正方形
        if h != w:
            max_dim = max(h, w)
            padded = np.full((max_dim, max_dim), float(cfg.hu_min), dtype=slice_2d.dtype)
            pad_y = (max_dim - h) // 2
            pad_x = (max_dim - w) // 2
            padded[pad_y:pad_y + h, pad_x:pad_x + w] = slice_2d
            slice_2d = padded
            h, w = max_dim, max_dim

        zoom_factor = target_size / h
        return zoom(slice_2d, zoom_factor, order=1, mode="nearest")

    # ---------------------------------------------------------------- #
    # 6. 窗位变换
    # ---------------------------------------------------------------- #
    def apply_window(self, volume: np.ndarray, wl: int, ww: int) -> np.ndarray:
        """
        应用窗宽窗位并归一化到 [0, 255] uint8

        CT 窗位变换公式:
          min_hu = WL - WW/2
          max_hu = WL + WW/2
          pixel  = (HU - min_hu) / WW * 255
        """
        min_v = wl - ww / 2.0
        max_v = wl + ww / 2.0
        clipped = np.clip(volume, min_v, max_v)
        normalized = (clipped - min_v) / (max_v - min_v) * 255.0
        return normalized.astype(np.uint8)

    # ---------------------------------------------------------------- #
    # 7. 多窗位通道合成 (可选的高级模式)
    # ---------------------------------------------------------------- #
    def merge_windows_to_rgb(
        self,
        volume: np.ndarray,
        windows: List[WindowConfig],
    ) -> np.ndarray:
        """
        将最多 3 个窗位分别映射到 R/G/B 通道, 生成伪彩色图像

        优点: 单张图像同时包含多窗位信息, 减少推理次数
        缺点: VLM 可能不擅长解读伪彩色编码, 需要在 prompt 中说明

        volume: (N, H, W) float32, HU 值
        windows: 最多 3 个 WindowConfig

        返回: (N, H, W, 3) uint8
        """
        assert len(windows) <= 3, "RGB 合成最多 3 个窗位"
        n, h, w = volume.shape
        rgb = np.zeros((n, h, w, 3), dtype=np.uint8)
        for ch_idx, win in enumerate(windows):
            rgb[..., ch_idx] = self.apply_window(volume, win.wl, win.ww)
        # 如果不足 3 个窗位, 剩余通道保持 0
        return rgb

    # ---------------------------------------------------------------- #
    # 主流程
    # ---------------------------------------------------------------- #
    def load_and_preprocess(
        self, file_path: str
    ) -> Tuple[Dict[str, np.ndarray], PreprocessMetadata]:
        """
        完整预处理流程

        返回:
          tensors_dict: {window_name: np.ndarray (N, H, W, 3) uint8}
                        每个窗位一个 3 通道数组 (灰度复制到 3 通道)
          metadata:     PreprocessMetadata, 包含空间信息可注入 prompt
        """
        logger.info(f"加载 NIfTI: {file_path}")
        ct_img = nib.load(file_path)

        # Step 1: 方向标准化 → (Z, Y, X) float32
        volume, spacing = self.standardize_orientation(ct_img)
        original_shape = tuple(volume.shape)
        original_spacing = spacing  # 保留原始 spacing, 用于元数据
        dz, dy, dx = spacing
        logger.info(
            f"方向标准化完成: shape={original_shape}, "
            f"spacing=({dz:.2f}, {dy:.2f}, {dx:.2f}) mm"
        )

        # Step 2: HU 值清洗
        volume = self.sanitize_hu(volume)

        # Step 2.5: XY 面内间距归一化 (借鉴 CT-CLIP)
        # 在 HU 清洗后立即归一化, 让后续所有步骤都在统一物理分辨率下进行
        # 这保证"病灶大小 8mm"等尺寸判断跨病例一致
        volume, spacing = self.normalize_xy_spacing(volume, spacing)
        normalized_shape = tuple(volume.shape)

        # Step 3: 自适应裁剪
        volume, crop_range = self.get_valid_anatomical_region(volume)
        cropped_depth = volume.shape[0]

        # Step 4: 切片采样
        sampled_indices = self.sample_slice_indices(
            cropped_depth,
            volume if cfg.sampling_strategy == "density" else None,
        )
        sampled_volume = volume[sampled_indices]  # (N, Y, X)

        # 【修复】：用空气值填充不足的切片，防止模型幻觉
        N_real = sampled_volume.shape[0]
        if N_real < self.target_slices:
            pad_shape = (self.target_slices - N_real, sampled_volume.shape[1], sampled_volume.shape[2])
            padding_vol = np.full(pad_shape, float(cfg.hu_min), dtype=sampled_volume.dtype)
            sampled_volume = np.concatenate([sampled_volume, padding_vol], axis=0)

            # 补齐层号映射
            original_indices = [idx + crop_range[0] for idx in sampled_indices]
            original_indices.extend([""] * (self.target_slices - N_real))  # 填充空白层号
        else:
            original_indices = [idx + crop_range[0] for idx in sampled_indices]

        # Step 5: 单切片 resize 到 cfg.slice_size
        N = sampled_volume.shape[0]
        resized = np.zeros((N, cfg.slice_size, cfg.slice_size), dtype=np.float32)
        for i in range(N):
            resized[i] = self.resize_slice(sampled_volume[i], cfg.slice_size)

        # Step 6: 多窗位变换 → 3 通道
        tensors_dict: Dict[str, np.ndarray] = {}
        for w in cfg.enabled_windows:
            windowed = self.apply_window(resized, w.wl, w.ww)     # (N, H, W) uint8
            three_channel = np.stack([windowed] * 3, axis=-1)      # (N, H, W, 3)
            tensors_dict[w.name] = three_channel

        # 构建元数据
        scan_coverage = cropped_depth * spacing[0]
        metadata = PreprocessMetadata(
            file_path=file_path,
            original_shape=original_shape,
            original_spacing_mm=original_spacing,
            normalized_shape=normalized_shape,
            normalized_spacing_mm=spacing,
            cropped_z_range=crop_range,
            cropped_depth=cropped_depth,
            sampled_z_indices=sampled_indices,
            sampled_z_indices_original=original_indices,
            effective_slice_spacing_mm=round(effective_spacing, 2),
            target_slices=self.target_slices,
            sampling_strategy=cfg.sampling_strategy,
            windows_used=[w.name for w in cfg.enabled_windows],
            slice_size=cfg.slice_size,
            tiled_image_size=cfg.tiled_image_size,
            scan_coverage_mm=round(scan_coverage, 1),
            xy_spacing_normalized=cfg.normalize_xy_spacing,
        )

        logger.info(
            f"预处理完成: 采样 {N} 层, 有效层间距 {effective_spacing:.1f}mm, "
            f"覆盖 {scan_coverage:.0f}mm, 窗位 {metadata.windows_used}"
        )
        return tensors_dict, metadata