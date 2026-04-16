"""
全局配置中心

设计原则:
  1. 拼图尺寸与 VLM 编码器的 patch 机制对齐
     - Qwen2.5-VL 视觉编码器使用 14×14 patch, 支持动态分辨率
     - VoxRep 论文核心思路: 每张切片 resize 到 patch 对齐尺寸, 再拼成网格
     - 我们选择 slice_size=224 (224/14=16 patches), grid=8×8=64 层
       拼图总尺寸 224*8=1792, Qwen2.5-VL 可处理此分辨率
  2. 多窗位设计覆盖全部位需求 (胸/腹/头/骨/纵隔)
  3. 所有超参数集中管理, 便于消融实验
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ================================================================== #
# 窗位配置
# ================================================================== #
@dataclass
class WindowConfig:
    """CT 窗宽窗位配置"""
    name: str       # 窗位名称, 也用于 prompt 路由
    wl: int         # 窗位 (Window Level, HU)
    ww: int         # 窗宽 (Window Width, HU)
    priority: int = 0  # 显示优先级, 越大越先处理


# 预定义窗位库
WINDOW_PRESETS = {
    "soft":        WindowConfig("soft",        40,   400, priority=10),
    "lung":        WindowConfig("lung",       -600, 1500, priority=9),
    "bone":        WindowConfig("bone",        400, 1800, priority=7),
    "mediastinum": WindowConfig("mediastinum",  50,  350, priority=8),
    "brain":       WindowConfig("brain",        40,   80, priority=6),
    "liver":       WindowConfig("liver",        60,  160, priority=5),
    "stroke":      WindowConfig("stroke",       32,    8, priority=4),
    "subdural":    WindowConfig("subdural",     75,  215, priority=3),
}


# ================================================================== #
# 主配置
# ================================================================== #
@dataclass
class Config:
    """全局配置, 所有模块从此处读取参数"""

    # ---- 模型 ---- #
    model_id: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    max_new_tokens: int = 4096
    temperature: float = 0.3       # CT 报告需要稳定输出, 低温
    top_p: float = 0.9
    use_multi_image: bool = True   # True: 每张窗位拼图作为独立 image token
                                   # False: 所有窗位拼到一张巨图(不推荐)

    # ---- 切片采样 ---- #
    target_slices: int = 64        # 采样层数, 须等于 grid_rows * grid_cols
    grid_rows: int = 8             # 拼图行数
    grid_cols: int = 8             # 拼图列数
    sampling_strategy: str = "uniform"  # uniform | weighted | density

    # ---- 切片预处理 ---- #
    slice_size: int = 224          # 单切片 resize 目标尺寸 (与 VLM patch 14×14 对齐)
    # 拼图总尺寸 = slice_size * grid_cols = 224 * 8 = 1792
    # Qwen2.5-VL 动态分辨率可处理, 无需额外 resize

    # ---- 体素间距归一化 (借鉴 CT-CLIP) ---- #
    # CT-CLIP 的做法: 所有 volume 重采样到 0.75×0.75×1.5mm 固定间距 + center-crop 到 480×480×240
    # 我们的轻量版: 只做 XY 面内间距归一化 + center-crop 到固定 XY 尺寸,
    # Z 轴交给切片采样策略处理 (避免 3D 重采样的插值伪影)
    # 目的: 保证跨病例的"肺结节 8mm"类尺寸判断一致 (不同扫描仪 XY 间距差异很大)
    normalize_xy_spacing: bool = True           # 是否做 XY 面内间距归一化
    target_xy_spacing_mm: float = 0.75          # 目标面内分辨率 (mm), 与 CT-CLIP 对齐
    target_xy_size: int = 480                   # 重采样后的目标面内尺寸 (CT-CLIP 使用 480×480)
    # 注: 重采样后的体积会再经过 resize_slice 缩放到 slice_size (224)
    #     所以 target_xy_size 的主要作用是先做物理间距对齐, 再统一到 VLM 输入尺寸

    # ---- HU 值范围 ---- #
    hu_min: float = -1024.0        # 空气
    hu_max: float = 3071.0         # 骨骼/金属上限

    # ---- 自适应裁剪 ---- #
    crop_threshold_ratio: float = 0.05   # 动态阈值 = min + (max-min) * ratio
    crop_intensity_percentile: int = 75  # 取切片第 N 分位数判断有效性
    crop_margin_slices: int = 2          # 裁剪边界额外保留的层数

    # ---- 窗位选择 ---- #
    # 根据扫描部位选择窗位组合, 默认使用全窗位
    enabled_window_names: List[str] = field(
        default_factory=lambda: ["soft", "lung", "bone", "mediastinum"]
    )

    # ---- 拼图可视化 ---- #
    border_size: int = 1           # 切片间分隔线宽度 (像素)
    border_value: int = 128        # 分隔线灰度 (中性灰, 降低模型注意力)
    draw_slice_index: bool = True  # 是否在切片上标注层号
    index_font_size: int = 12      # 层号字体大小

    # ---- 输出 ---- #
    output_dir: str = "outputs"
    save_debug_images: bool = True   # 保存拼图用于人工检查
    save_metadata: bool = True       # 保存预处理元数据 JSON

    # ---- 派生属性 ---- #
    @property
    def enabled_windows(self) -> List[WindowConfig]:
        """按 priority 降序返回启用的窗位配置"""
        windows = []
        for name in self.enabled_window_names:
            if name in WINDOW_PRESETS:
                windows.append(WINDOW_PRESETS[name])
            else:
                raise ValueError(f"未知窗位名称: {name}, 可选: {list(WINDOW_PRESETS.keys())}")
        return sorted(windows, key=lambda w: w.priority, reverse=True)

    @property
    def tiled_image_size(self) -> Tuple[int, int]:
        """拼图总尺寸 (H, W)"""
        return (self.slice_size * self.grid_rows, self.slice_size * self.grid_cols)

    def validate(self):
        """启动时校验配置一致性"""
        assert self.target_slices == self.grid_rows * self.grid_cols, \
            f"target_slices({self.target_slices}) != grid({self.grid_rows}x{self.grid_cols})"
        assert self.slice_size % 14 == 0, \
            f"slice_size({self.slice_size}) 应为 14 的倍数以对齐 VLM patch"
        assert len(self.enabled_window_names) > 0, "至少启用一个窗位"


# 全局单例
cfg = Config()
cfg.validate()