"""
CT 预处理核心模块
负责: NIfTI 读取 -> 方向标准化 -> 数据清洗 -> 自适应裁剪 -> 切片采样 -> 多窗位归一化 -> 单切片 resize
输出: 字典 {window_name: np.ndarray of shape (N, H, W, 3) uint8} + 预处理元数据
"""
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
import nibabel.orientations as ornt
from scipy.ndimage import zoom

from configs.config import cfg, WindowConfig

logger = logging.getLogger(__name__)


@dataclass
class PreprocessMetadata:
    """预处理过程的元数据，便于溯源和注入 prompt"""
    file_path: str
    original_shape: Tuple[int, int, int]      # (Z, Y, X) after orientation
    original_spacing: Tuple[float, float, float]  # (dz, dy, dx) in mm
    cropped_z_range: Tuple[int, int]          # 自适应裁剪后保留的 Z 范围
    sampled_z_indices: List[int]              # 实际抽样使用的层号(基于裁剪后)
    target_slices: int
    sampling_strategy: str
    windows_used: List[str]
    slice_size: int

    def to_dict(self):
        return asdict(self)


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
        放射学惯例: Z 轴自下而上, Y 轴由前到后, X 轴由右到左
        同时返回 (dz, dy, dx) 体素间距, 单位 mm
        """
        orig_ornt = nib.io_orientation(ct_img.affine)
        targ_ornt = ornt.axcodes2ornt(("L", "P", "S"))
        transform = ornt.ornt_transform(orig_ornt, targ_ornt)
        aligned = ornt.apply_orientation(ct_img.get_fdata(), transform)

        # zooms 顺序与原 affine 对应, 这里需要按 transform 重新映射
        # 简化处理: 使用 header 提供的 zooms 并按 (Z, Y, X) 顺序取
        zooms = ct_img.header.get_zooms()[:3]
        # apply_orientation 后的轴顺序对应 LPS,转置 (2,1,0) -> (S, P, L) = (Z, Y, X)
        volume = np.transpose(aligned, (2, 1, 0)).astype(np.float32)
        spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
        return volume, spacing

    # ---------------------------------------------------------------- #
    # 2. 数据清洗: NaN / Inf / HU 越界
    # ---------------------------------------------------------------- #
    def sanitize_hu(self, volume: np.ndarray) -> np.ndarray:
        """处理 NaN/Inf 并把 HU 值限制在物理合理区间"""
        if not np.all(np.isfinite(volume)):
            n_bad = np.sum(~np.isfinite(volume))
            logger.warning(f"检测到 {n_bad} 个 NaN/Inf 体素, 替换为 hu_min")
            volume = np.nan_to_num(volume, nan=cfg.hu_min,
                                   posinf=cfg.hu_max, neginf=cfg.hu_min)
        return np.clip(volume, cfg.hu_min, cfg.hu_max)

    # ---------------------------------------------------------------- #
    # 3. 自适应裁剪: 去除头尾空气层和扫描床外区域
    # ---------------------------------------------------------------- #
    def get_valid_anatomical_region(
        self, volume: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """基于强度分位数判断"是否包含人体组织", 裁掉头尾纯空气层"""
        depth = volume.shape[0]
        global_min = float(np.min(volume))
        global_max = float(np.max(volume))
        dyn_threshold = global_min + (global_max - global_min) * cfg.crop_threshold_ratio

        valid = []
        for z in range(depth):
            if np.percentile(volume[z], cfg.crop_intensity_percentile) > dyn_threshold:
                valid.append(z)

        if not valid:
            logger.warning("自适应裁剪未找到有效层, 保留全卷")
            return volume, (0, depth - 1)

        start_z, end_z = valid[0], valid[-1]
        logger.info(f"自适应裁剪 Z 轴: [{start_z}:{end_z}] (原始深度 {depth})")
        return volume[start_z:end_z + 1], (start_z, end_z)

    # ---------------------------------------------------------------- #
    # 4. 切片采样: 决定哪 N 层进入拼图
    # ---------------------------------------------------------------- #
    def sample_slice_indices(self, depth: int) -> List[int]:
        """
        根据 sampling_strategy 返回采样的层 index 列表(基于裁剪后的 volume)
          - all/uniform: 均匀采样, 首尾必含
          - weighted:    中段(20%~80%)加权, 边缘也保留少量
        如果 depth <= target_slices, 全部保留 + 末尾重复补齐
        """
        target = self.target_slices

        # Case 1: 切片数不足, 全用 + 重复末张补齐(避免插值伪影)
        if depth <= target:
            indices = list(range(depth))
            while len(indices) < target:
                indices.append(depth - 1)
            return indices

        strategy = cfg.sampling_strategy
        if strategy in ("uniform", "all"):
            # np.linspace 含端点, 保证首尾切片必被采样
            return [int(round(i)) for i in np.linspace(0, depth - 1, target)]

        if strategy == "weighted":
            # 边缘各 15%, 中段 70%
            n_top = max(1, int(target * 0.15))
            n_bot = max(1, int(target * 0.15))
            n_mid = target - n_top - n_bot
            z1 = int(depth * 0.2)
            z2 = int(depth * 0.8)
            idx_top = np.linspace(0, z1 - 1, n_top, dtype=int) if z1 > 0 else []
            idx_mid = np.linspace(z1, z2 - 1, n_mid, dtype=int) if z2 > z1 else []
            idx_bot = np.linspace(z2, depth - 1, n_bot, dtype=int) if depth > z2 else []
            indices = sorted(set(list(idx_top) + list(idx_mid) + list(idx_bot)))
            # 去重后可能不够,用均匀补齐
            if len(indices) < target:
                fill = np.linspace(0, depth - 1, target, dtype=int).tolist()
                for v in fill:
                    if v not in indices:
                        indices.append(v)
                    if len(indices) >= target:
                        break
                indices = sorted(indices)[:target]
            return indices

        raise ValueError(f"未知采样策略: {strategy}")

    # ---------------------------------------------------------------- #
    # 5. 单切片 resize 到统一尺寸
    # ---------------------------------------------------------------- #
    def resize_slice(self, slice_2d: np.ndarray, target_size: int) -> np.ndarray:
        """对单张 (H, W) 切片进行双线性插值 resize 到 (target_size, target_size)"""
        h, w = slice_2d.shape
        if (h, w) == (target_size, target_size):
            return slice_2d
        zoom_factors = (target_size / h, target_size / w)
        # order=1 双线性, 适合 CT 这种连续灰度
        return zoom(slice_2d, zoom_factors, order=1, mode="nearest")

    # ---------------------------------------------------------------- #
    # 6. 窗位变换 (HU -> uint8)
    # ---------------------------------------------------------------- #
    def apply_window(self, volume: np.ndarray, wl: int, ww: int) -> np.ndarray:
        """应用窗宽窗位并归一化到 8-bit"""
        min_v = wl - ww / 2
        max_v = wl + ww / 2
        clipped = np.clip(volume, min_v, max_v)
        return ((clipped - min_v) / ww * 255.0).astype(np.uint8)

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
          metadata:     PreprocessMetadata
        """
        logger.info(f"加载 NIfTI: {file_path}")
        ct_img = nib.load(file_path)

        # Step 1: 方向标准化
        volume, spacing = self.standardize_orientation(ct_img)
        original_shape = tuple(volume.shape)

        # Step 2: HU 值清洗
        volume = self.sanitize_hu(volume)

        # Step 3: 自适应裁剪
        volume, crop_range = self.get_valid_anatomical_region(volume)

        # Step 4: 切片采样
        sampled_indices = self.sample_slice_indices(volume.shape[0])
        sampled_volume = volume[sampled_indices]   # (N, Y, X)

        # Step 5: 单切片 resize 到统一尺寸
        N = sampled_volume.shape[0]
        resized = np.zeros((N, cfg.slice_size, cfg.slice_size), dtype=np.float32)
        for i in range(N):
            resized[i] = self.resize_slice(sampled_volume[i], cfg.slice_size)

        # Step 6: 多窗位变换 + 通道扩展(VLM 期望 3 通道)
        tensors_dict: Dict[str, np.ndarray] = {}
        for w in cfg.enabled_windows:
            windowed = self.apply_window(resized, w.wl, w.ww)        # (N, H, W) uint8
            three_channel = np.stack([windowed] * 3, axis=-1)         # (N, H, W, 3)
            tensors_dict[w.name] = three_channel

        metadata = PreprocessMetadata(
            file_path=file_path,
            original_shape=original_shape,
            original_spacing=spacing,
            cropped_z_range=crop_range,
            sampled_z_indices=sampled_indices,
            target_slices=self.target_slices,
            sampling_strategy=cfg.sampling_strategy,
            windows_used=[w.name for w in cfg.enabled_windows],
            slice_size=cfg.slice_size,
        )
        return tensors_dict, metadata
