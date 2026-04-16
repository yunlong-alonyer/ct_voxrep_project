"""
业务总调度
流程: 预处理 -> Map(各窗位独立阅片) -> Reduce(LLM 融合报告)

注意: 本脚本生成的报告仅用于研究/微调阶段的 baseline 对照,
      不可作为临床诊断依据。
"""
import os
import logging
from pathlib import Path

from configs.config import cfg
from src.data_processor import CTProcessor
from src.utils import convert_voxel_to_2d_scan_image, save_2d_image, save_metadata
from src.model_engine import QwenVLEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ====================================================================== #
# Prompt 模板
# ====================================================================== #
WINDOW_PROMPTS = {
    "soft": (
        "这是一张CT软组织窗的Z轴序列拼图,按从足侧到头侧顺序排列(8x8网格,共64层),"
        "每张子图左上角的数字为该层在原始扫描中的层号。"
        "请以放射科医生视角,系统描述肝、脾、胰、肾、肾上腺、淋巴结及腹膜后等"
        "实质性脏器与软组织的形态、密度及强化情况;若发现占位性病变,"
        "请描述其位置、大小、边界、密度特征。忽略骨骼与肺气描述。"
    ),
    "lung": (
        "这是一张CT肺窗的Z轴序列拼图,按从足侧到头侧顺序排列(8x8网格,共64层),"
        "每张子图左上角的数字为该层在原始扫描中的层号。"
        "请详细描述双肺实质、支气管、胸膜及胸腔情况,"
        "明确指出有无结节(注明大小及位置)、磨玻璃影、实变、纤维条索、"
        "胸腔积液或气胸。忽略骨骼与腹部脏器。"
    ),
    "bone": (
        "这是一张CT骨窗的Z轴序列拼图,按从足侧到头侧顺序排列(8x8网格,共64层),"
        "每张子图左上角的数字为该层在原始扫描中的层号。"
        "请详细描述椎体、肋骨、胸骨、骨盆等可见骨骼结构,"
        "包括骨皮质连续性、骨髓腔密度、关节间隙;"
        "明确指出有无骨折、骨质破坏、骨质增生或硬化灶。忽略软组织与肺。"
    ),
    "mediastinum": (
        "这是一张CT纵隔窗的Z轴序列拼图,按从足侧到头侧顺序排列(8x8网格,共64层)。"
        "请描述心脏大小形态、大血管(主动脉、肺动脉)、气管隆突、"
        "纵隔淋巴结及食管情况;指出有无肿大淋巴结(短径>1cm)或异常密度灶。"
    ),
    "brain": (
        "这是一张CT脑窗的Z轴序列拼图,按从足侧到头侧顺序排列(8x8网格,共64层)。"
        "请描述脑实质密度、脑室系统、脑沟脑池、中线结构,"
        "指出有无出血、梗死、占位或脑水肿。"
    ),
}


REDUCE_PROMPT_TEMPLATE = """你是一名资深的影像科主任医师。以下是同一份CT扫描在不同窗位下的独立阅片记录,
每份记录由初级医师在仅观察单一窗位的条件下完成,可能存在描述重叠或遗漏。

{window_reports}

请你执行以下任务:
1. 整合所有窗位发现,剔除重复描述,保留特异性最强的窗位的判断(肺部病变以肺窗为准,
   骨骼病变以骨窗为准,实质脏器病变以软组织/纵隔窗为准);
2. 对存在矛盾的描述,标注"需复核"而非武断取舍;
3. 撰写一份结构化放射学报告,严格遵循以下章节:

【影像表现】
按解剖部位分段描述(胸部 / 腹部 / 骨骼系统等)。

【影像诊断】
列出主要诊断与次要诊断,每条诊断后注明对应的影像学依据。

【建议】
针对每条诊断给出后续检查或随访建议。

【局限性说明】
本报告由AI模型基于64层采样切片生成,可能遗漏小于采样间距的微小病灶,
最终诊断须由具备资质的放射科医师结合原始薄层影像复核。
"""


# ====================================================================== #
# 主流程
# ====================================================================== #
def run_pipeline(ct_file_path: str) -> str:
    if not os.path.exists(ct_file_path):
        raise FileNotFoundError(f"找不到 CT 文件: {ct_file_path}")

    case_name = Path(ct_file_path).stem.replace(".nii", "")
    case_output_dir = os.path.join(cfg.output_dir, case_name)
    os.makedirs(case_output_dir, exist_ok=True)

    # ---------- Stage 1: 预处理 ----------
    logger.info("=" * 60)
    logger.info("Stage 1: CT 预处理")
    logger.info("=" * 60)
    processor = CTProcessor()
    tensors_dict, metadata = processor.load_and_preprocess(ct_file_path)

    if cfg.save_metadata:
        save_metadata(metadata.to_dict(), case_output_dir, "preprocess_meta.json")

    # ---------- Stage 2: 拼图重组 ----------
    logger.info("=" * 60)
    logger.info("Stage 2: 拼图重组")
    logger.info("=" * 60)
    tiled_images = {}
    for win_name, vol in tensors_dict.items():
        tiled = convert_voxel_to_2d_scan_image(
            ct_volume=vol,
            grid_rows=cfg.grid_rows,
            grid_cols=cfg.grid_cols,
            border_size=cfg.border_size,
            border_value=cfg.border_value,
            draw_index=cfg.draw_slice_index,
            z_indices=metadata.sampled_z_indices_original,
        )
        tiled_images[win_name] = tiled
        if cfg.save_debug_images:
            save_2d_image(tiled, case_output_dir, f"tiled_{win_name}.jpg")

    # ---------- Stage 3: Map ----------
    logger.info("=" * 60)
    logger.info("Stage 3: Map 阶段 - 各窗位独立阅片")
    logger.info("=" * 60)
    engine = QwenVLEngine()
    sub_reports = {}
    for win_name, tiled in tiled_images.items():
        prompt = WINDOW_PROMPTS.get(
            win_name,
            f"这是一张CT {win_name}窗的Z轴序列拼图,请描述所见。",
        )
        logger.info(f"-> 处理 {win_name} 窗 ...")
        sub_reports[win_name] = engine.predict_multi_image([tiled], prompt)
        logger.info(f"[{win_name}] 子报告生成完成")

    # ---------- Stage 4: Reduce ----------
    logger.info("=" * 60)
    logger.info("Stage 4: Reduce 阶段 - 报告融合")
    logger.info("=" * 60)
    window_reports_str = "\n\n".join(
        f"【{name.upper()} 窗记录】\n{text}" for name, text in sub_reports.items()
    )
    reduce_prompt = REDUCE_PROMPT_TEMPLATE.format(window_reports=window_reports_str)
    final_report = engine.predict_text(reduce_prompt)

    # 保存最终报告
    report_path = os.path.join(case_output_dir, "final_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# CT 影像报告 — {case_name}\n\n")
        f.write(final_report)
        f.write("\n\n---\n## 附录: 各窗位原始阅片\n")
        for name, text in sub_reports.items():
            f.write(f"\n### {name.upper()} 窗\n{text}\n")
    logger.info(f"最终报告已保存: {report_path}")

    return final_report


def main():
    ct_file_path = "sample_data/patient_001.nii.gz"   # 替换为实际路径
    final = run_pipeline(ct_file_path)
    print("\n" + "═" * 60)
    print("🏥 AI 生成放射学诊断报告")
    print("═" * 60)
    print(final)


if __name__ == "__main__":
    main()
