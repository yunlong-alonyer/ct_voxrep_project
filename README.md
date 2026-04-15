# CT-VoxRep 项目

基于 VoxRep 思路改造的 CT 影像 → Qwen3-VL 报告生成预处理流水线。
**当前阶段定位**：为后续微调提供干净、稳定、可溯源的输入数据；尚未微调时输出仅供研究对照，不可临床使用。

---

## 目录结构

```
ct_voxrep_project/
├── requirements.txt
├── main.py                       # 业务总调度
├── README.md
│
├── configs/
│   ├── __init__.py
│   └── config.py                 # 全局配置中心(网格/窗位/采样策略等)
│
├── src/
│   ├── __init__.py
│   ├── data_processor.py         # CT 预处理流水线
│   ├── utils.py                  # 拼图重组与可视化
│   └── model_engine.py           # Qwen3-VL 推理封装
│
├── sample_data/                  # 放置 .nii.gz 原始 CT 文件
└── output/                       # 每个 case 一个子目录,含拼图/元数据/报告
    └── <case_name>/
        ├── preprocess_meta.json
        ├── tiled_soft.jpg
        ├── tiled_lung.jpg
        ├── tiled_bone.jpg
        └── final_report.md
```

---

## 相比原方案的关键改进

| 维度 | 原方案 | 本方案 | 改进理由 |
|------|--------|--------|----------|
| 切片数 | 16 (4×4) | **64 (8×8)**, 可配置 | 16 层对薄层 CT 严重不足,易漏小病灶 |
| 切片尺寸 | 224 (固定) | **112, 可配置** | 64 层 × 112px = 896px,与 Qwen-VL 期望尺寸匹配,且不浪费 token |
| 切片间分隔 | 白边 (255) | **中性灰 (128) 或可关闭** | 白边形成强信号,干扰模型对解剖结构的注意力 |
| 层号标注 | 无 | **每张切片左上角叠加 Z 轴层号** | 帮助模型建立 Z 轴顺序感,微调时可在 prompt 中引用 |
| 数据清洗 | 无 | **NaN/Inf 替换 + HU 越界裁剪** | 部分 NIfTI 文件含伪影,直接 clip 会导致归一化崩溃 |
| 单切片尺寸归一化 | 无 | **统一双线性 resize** | 处理非正方形扫描(如 512×384) |
| 采样策略 | 固定 20/60/20 中段加权 | **uniform / weighted / all 三选一** | 不同部位扫描病灶分布不同,中段加权易遗漏边界 |
| 窗位 | soft/lung/bone 固定 | **5 种窗位可独立启用** | 增加纵隔窗与脑窗,适配不同部位 CT |
| 元数据 | 无 | **预处理元数据 JSON 落盘** | 便于微调阶段注入 prompt、便于调试溯源 |
| 输出组织 | 单一目录 | **每 case 一个子目录** | 批量处理时不会互相覆盖 |

---

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 放入数据
将 `.nii` 或 `.nii.gz` 文件放到 `sample_data/`。

### 3. 调整配置
编辑 `configs/config.py`：
- 显存紧张时把 `grid_rows/grid_cols` 调到 6×6 或 4×4
- 部位为头部时启用 `brain` 窗、关闭 `lung` 窗
- 调试时可设 `border_size=0` 与 `draw_slice_index=False` 对比效果

### 4. 运行
```bash
python main.py
```

输出会在 `output/<case_name>/` 下，包含：
- 三张拼图 JPG（人工核查用）
- 预处理元数据 JSON
- 最终融合报告 Markdown

---

## 微调阶段衔接建议

1. **训练数据生成**：用本流水线对训练集 CT 批量预处理，得到 (拼图, 元数据, 真实报告) 三元组
2. **Prompt 模板**：把元数据中的 `sampled_z_indices` 和 `original_spacing` 注入 prompt，让模型理解每张切片的物理位置
3. **数据增强**：可在 `data_processor.py` 中添加随机窗宽窗位扰动、随机采样起点偏移
4. **消融实验**：通过修改 config 即可对比不同网格、不同分隔线方案的效果，无需改动核心代码

---

## 已知局限（必读）

1. **64 层仍可能漏诊小于 5mm 的病灶**：如需更高敏感度，需用滑窗多次推理（未实现，可在 `main.py` 中扩展）
2. **拼图破坏严格 Z 轴连续性**：模型只能跨 patch 推理而非真 3D 卷积，对需要追踪的结构（小血管、细支气管）有限
3. **未微调前 Qwen3-VL 在医学影像上有明确幻觉风险**：本流水线只解决"输入侧"，幻觉问题靠后续微调
4. **未做病灶定位 grounding**：模型只输出文字描述，不输出坐标框；如需 grounding，需要在微调阶段加入定位监督信号
