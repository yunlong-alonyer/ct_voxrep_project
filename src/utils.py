"""
将多张 2D 切片重组为单张拼图，并提供可视化与元数据保存
"""
import os
import json
import logging
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from configs.config import cfg

logger = logging.getLogger(__name__)


def _get_font(size: int = 10) -> Optional[ImageFont.ImageFont]:
    """尝试加载小字体, 失败则返回 PIL 默认位图字体"""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def convert_voxel_to_2d_scan_image(
    ct_volume: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    border_size: int = 1,
    border_value: int = 128,
    draw_index: bool = True,
    z_indices: Optional[list] = None,
) -> np.ndarray:
    """
    将 (N, H, W, 3) uint8 的切片堆叠成 (Hc, Wc, 3) uint8 拼图

    参数:
      ct_volume:   (N, H, W, 3) 切片数组, N 应等于 grid_rows*grid_cols
      border_size: 切片间分隔线宽度,0 表示不画
      border_value:分隔线灰度,推荐 128(中性灰)而非 255(白)以减弱模型对边框的注意力
      draw_index:  是否在每张切片左上角写入 Z 轴层号
      z_indices:   实际层号列表(用于标注),长度应与 N 一致

    返回:
      (image_h, image_w, 3) uint8 numpy 数组
    """
    assert ct_volume.ndim == 4 and ct_volume.shape[-1] == 3, \
        f"输入应为 (N, H, W, 3) uint8, 实际 {ct_volume.shape}"

    n, h, w, c = ct_volume.shape
    expected = grid_rows * grid_cols
    assert n == expected, f"切片数 {n} 与网格 {grid_rows}x{grid_cols}={expected} 不匹配"

    patch_h = h + 2 * border_size
    patch_w = w + 2 * border_size
    image_h = patch_h * grid_rows
    image_w = patch_w * grid_cols

    canvas = np.zeros((image_h, image_w, c), dtype=np.uint8)

    for i in range(n):
        row = i // grid_cols
        col = i % grid_cols
        y0 = row * patch_h
        x0 = col * patch_w

        # 绘制分隔线(中性灰,降低边框对模型的干扰)
        if border_size > 0:
            patch_area = canvas[y0:y0 + patch_h, x0:x0 + patch_w]
            patch_area[:border_size, :] = border_value
            patch_area[-border_size:, :] = border_value
            patch_area[:, :border_size] = border_value
            patch_area[:, -border_size:] = border_value

        # 填入切片内容
        canvas[y0 + border_size: y0 + border_size + h,
               x0 + border_size: x0 + border_size + w] = ct_volume[i]

    # 叠加层号标注(可选)
    if draw_index:
        pil_img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_img)
        font = _get_font(size=max(10, h // 12))
        for i in range(n):
            row = i // grid_cols
            col = i % grid_cols
            label = str(z_indices[i]) if z_indices is not None else str(i)
            tx = col * patch_w + border_size + 2
            ty = row * patch_h + border_size + 1
            # 黑色描边 + 黄色文字, 在任何窗位下都清晰
            draw.text((tx, ty), label, fill=(255, 255, 0),
                      font=font, stroke_width=1, stroke_fill=(0, 0, 0))
        canvas = np.array(pil_img)

    return canvas


def save_2d_image(image_np: np.ndarray, save_dir: str, filename: str) -> str:
    """保存拼图用于可视化检查, 返回保存路径"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    Image.fromarray(image_np.astype(np.uint8)).save(save_path)
    logger.info(f"拼图已保存: {save_path}")
    return save_path


def save_metadata(metadata_dict: dict, save_dir: str, filename: str) -> str:
    """保存预处理元数据 JSON"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"元数据已保存: {save_path}")
    return save_path
