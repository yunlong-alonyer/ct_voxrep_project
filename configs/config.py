"""
全局配置中心
所有可调超参数集中管理，便于消融实验和微调阶段切换
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, List


@dataclass
class WindowConfig:
    """单个窗位的窗宽窗位定义"""
    name: str
    wl: int   # window level
    ww: int   # window width
    enabled: bool = True


@dataclass
class ProjectConfig:
    # ============ 拼图网格配置 ============
    # 8x8 = 64 张切片，相比原方案的 16 张大幅提升 Z 轴覆盖率
    # 对薄层 CT (1-2mm) 而言，64 层仍属保守，但已能覆盖大多数中等大小病灶
    grid_rows: int = 8
    grid_cols: int = 8

    # 单切片在拼图中的边长（像素）。Qwen3-VL 支持动态分辨率，
    # 单切片 112px × 8 行 = 896px，与论文一致；可按显存调整
    slice_size: int = 112

    # 切片之间的分隔线宽度（像素）。0 表示不画分隔线
    border_size: int = 1
    # 分隔线灰度值（0-255）。用 128 中性灰，避免像 255 白边那样形成强信号
    border_value: int = 128

    # 是否在每个切片左上角叠加层号标注（帮助模型建立 Z 轴顺序感）
    draw_slice_index: bool = True

    # ============ CT 预处理配置 ============
    # 自适应裁剪的强度阈值百分位（0-100）。越高裁得越狠
    crop_intensity_percentile: float = 90.0
    # 判定为"有效层"的阈值占动态范围的比例
    crop_threshold_ratio: float = 0.1

    # HU 值合法范围（CT 物理上限）。超出此范围的体素视为伪影并裁剪
    hu_min: float = -1024.0
    hu_max: float = 3071.0

    # 采样策略: "uniform" (均匀) | "weighted" (中段加权) | "all" (不抽样,层数<=target时)
    sampling_strategy: str = "uniform"

    # ============ 多窗位配置 ============
    windows: List[WindowConfig] = field(default_factory=lambda: [
        WindowConfig(name="soft",       wl=40,    ww=400,  enabled=True),   # 腹部软组织
        WindowConfig(name="lung",       wl=-600,  ww=1500, enabled=True),   # 肺窗
        WindowConfig(name="bone",       wl=400,   ww=1500, enabled=True),   # 骨窗
        WindowConfig(name="mediastinum",wl=50,    ww=350,  enabled=False),  # 纵隔窗(可选)
        WindowConfig(name="brain",      wl=40,    ww=80,   enabled=False),  # 脑窗(可选)
    ])

    # ============ 模型与推理配置 ============
    model_id: str = "Qwen/Qwen3-VL-8B-Thinking"
    max_new_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95

    # ============ 路径配置 ============
    output_dir: str = "./output"
    save_debug_images: bool = True   # 是否保存拼图用于人工核查
    save_metadata: bool = True       # 是否保存预处理元数据 JSON

    @property
    def target_slices(self) -> int:
        return self.grid_rows * self.grid_cols

    @property
    def enabled_windows(self) -> List[WindowConfig]:
        return [w for w in self.windows if w.enabled]

    @property
    def composite_image_size(self) -> Tuple[int, int]:
        """拼图最终尺寸 (H, W)，含边框"""
        patch = self.slice_size + 2 * self.border_size
        return (patch * self.grid_rows, patch * self.grid_cols)

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


cfg = ProjectConfig()
