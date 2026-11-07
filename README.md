> 闲时重构，见重构分支。

# TCGA-STAD 多组学分型项目

本仓库用于完成“生物信息学实验课”胃腺癌（STAD）多组学分型实验。项目基于课程提供的 TCGA-STAD 数据，完成多组学整合分型、临床与生存评估、多模型共识、监督分类器复现、compact signature 筛选、结果导出以及正式 Word 报告生成。

当前代码主线和 Word 正式报告已经完成，后续主要工作是补齐真实成员信息并制作答辩 PPT。

最近更新：`2026-04-24`

- 补齐阶段 G 监督分类器中的 `MLP / LSTM / Transformer`。
- 重新导出 Word 正式报告：[docs/STAD多组学分型实验报告.docx](docs/STAD多组学分型实验报告.docx)。
- 全量测试通过：`28 passed`。

## 当前结论

- 主发现队列：5 组学完整交集，`mutation + cnv + methylation + expression + mirna`，`364` 个样本，SNF baseline 最优 `K=2`。
- 扩展验证队列：6 组学完整交集，在主发现队列基础上加入 `rppa`，`288` 个样本，SNF baseline 最优 `K=4`。
- 当前多模型共识方案：`resolution_guarded_copy_rescued`。
- 当前监督分类器：`Elastic Net`、`SVM`、`XGBoost`、`MLP`、tabular-sequence `LSTM`、tabular-sequence `Transformer`。
- 当前正式 final signature：`stable_reduced_10_cnv_burden_only`。
- final signature 10 个特征：`TMEM189`、`hsa-miR-139-3p`、`GOLM1`、`CNV burden`、`PROC`、`CDX1`、`hsa-miR-3065-3p`、`ARID1A`、`TMB`、`mutation event fraction`。
- 生存分析终点已覆盖 TCGA-CDR 推荐的 `OS / DSS / DFI / PFI`。

## 目录结构

- `biography_data/`: 当前正式使用的数据目录，包含已解压的 TCGA-STAD 多组学、临床、生存与探针注释文件。
- `src/stad_subtyping/`: 核心 Python 包，包括数据读取、预处理、SNF、聚类、评估、共识、分类器和 signature 逻辑。
- `scripts/run_full_pipeline.py`: 一键运行完整分析流程。
- `scripts/run_classifier_extension.py`: 在已有分类器输入缓存基础上单独补跑或重跑监督分类器。
- `scripts/export_report_docx.py`: 将 Markdown 实验报告导出为 Word 正式报告。
- `results/`: 当前正式结果目录，包含主发现队列结果、扩展验证队列结果、图件、分类器、signature 和解释表。
- `docs/`: 实验报告、执行计划、长期记忆、课程 PDF 和 Word 报告。
- `tests/`: 端到端与核心函数测试。

## 环境配置

推荐使用本地虚拟环境：

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

如果已经存在 `.venv`，可直接使用现有环境。

## 一键运行

当前推荐数据目录是 `biography_data/`：

```bash
.venv/bin/python scripts/run_full_pipeline.py \
  --data-dir biography_data \
  --out-dir results \
  --seed 2026
```

说明：

- 运行会重新生成 `results/` 下的主分析、扩展验证、对照模型、共识、分类器和 signature 结果。
- 全量运行耗时较长，日常修改文档或报告时不需要重复运行。
- 如果只想复现旧版 `data/` 目录结果，也可以将 `--data-dir` 改为 `data`，但当前正式口径以 `biography_data/` 为准。

## 单独补跑分类器

如果完整 pipeline 已经跑过，只想基于现有 `results/classifiers/elastic_net/*/feature_matrix.tsv` 和 `labels.tsv` 补跑监督模型，可使用：

```bash
.venv/bin/python scripts/run_classifier_extension.py \
  --results-dir results \
  --seed 2026 \
  --classifiers mlp lstm transformer
```

说明：

- 该脚本不会重做无监督分型，只更新指定分类器目录和 `results/classifier_summary.tsv`。
- `lstm` 与 `transformer` 是 tabular-sequence 对照模型：它们把多组学表格特征重排为伪序列后评估，不代表真正有时间序列或文本序列结构。
- 当前 final signature 的稳定筛选来源仍限定为 `Elastic Net / SVM / XGBoost`，新增深度对照模型不改写已定稿的 10 特征面板。

## 测试

```bash
.venv/bin/pytest -q
```

当前最近一次验证结果为：

```text
28 passed
```

## 关键输出

结果总览：

- [results/run_summary.json](results/run_summary.json)
- [results/queue_summary.tsv](results/queue_summary.tsv)
- [results/README.md](results/README.md)

主发现队列：

- [results/subtypes.tsv](results/subtypes.tsv)
- [results/k_selection.tsv](results/k_selection.tsv)
- [results/survival_stats.tsv](results/survival_stats.tsv)
- [results/clinical_assoc.tsv](results/clinical_assoc.tsv)
- [results/subtype_interpretation.tsv](results/subtype_interpretation.tsv)
- [results/figures/integrated_heatmap.png](results/figures/integrated_heatmap.png)

扩展验证队列：

- [results/validation_with_rppa/subtypes.tsv](results/validation_with_rppa/subtypes.tsv)
- [results/validation_with_rppa/k_selection.tsv](results/validation_with_rppa/k_selection.tsv)
- [results/validation_with_rppa/survival_stats.tsv](results/validation_with_rppa/survival_stats.tsv)
- [results/validation_with_rppa/clinical_assoc.tsv](results/validation_with_rppa/clinical_assoc.tsv)
- [results/validation_with_rppa/subtype_interpretation.tsv](results/validation_with_rppa/subtype_interpretation.tsv)
- [results/validation_with_rppa/figures/integrated_heatmap.png](results/validation_with_rppa/figures/integrated_heatmap.png)

模型、共识和分类器：

- [results/model_comparison.tsv](results/model_comparison.tsv)
- [results/model_evaluation_summary.tsv](results/model_evaluation_summary.tsv)
- [results/model_stability_summary.tsv](results/model_stability_summary.tsv)
- [results/sensitivity_overview.tsv](results/sensitivity_overview.tsv)
- [results/consensus_summary.tsv](results/consensus_summary.tsv)
- [results/classifier_summary.tsv](results/classifier_summary.tsv)

Final signature：

- [results/signature_final_recommended_panel_features.tsv](results/signature_final_recommended_panel_features.tsv)
- [results/signature_final_classifier_evidence.tsv](results/signature_final_classifier_evidence.tsv)
- [results/signature_final_interpretation.tsv](results/signature_final_interpretation.tsv)
- [results/signatures/final_signature_interpretation.md](results/signatures/final_signature_interpretation.md)

## 报告文件

Markdown 报告：

- [docs/STAD多组学分型实验报告.md](docs/STAD多组学分型实验报告.md)

Word 正式报告：

- [docs/STAD多组学分型实验报告.docx](docs/STAD多组学分型实验报告.docx)

当前 Word 版本已在 `2026-04-24 19:53 +0800` 重新导出，并确认包含 `MLP / LSTM / Transformer` 分类器结果。

重新导出 Word：

```bash
python scripts/export_report_docx.py
```

注意：

- 如需固定使用 `.venv` 导出，请先确认已执行 `.venv/bin/pip install -r requirements.txt`，并且 `.venv` 内可导入 `python-docx`。
- Word 封面和成员信息表中仍有“待填写”占位。
- 打开 Word 后建议右键更新目录。
- 提交前需要补真实抽签序号、课题组长、成员人数、作者、分工和贡献度。

## 方法概览

1. 样本统一到 12 位 TCGA patient ID，并仅保留原发肿瘤样本。
2. 各组学分别完成缺失值处理、特征筛选和标准化。
3. 连续型模态使用 PCA，离散型突变和 CNV 使用 TruncatedSVD。
4. 每个模态构建样本相似网络，并通过 SNF 融合。
5. 在 `K=2..4` 范围内进行共识聚类和谱聚类。
6. 输出临床关联、KM 生存曲线、log-rank 检验、多变量 Cox、TCGA-like 启发式映射和 subtype 解释表。
7. 通过 `SNF / early integration / NMF` 做无监督模型对照。
8. 通过 copy-rescued 多模型共识生成正式共识标签。
9. 使用 `Elastic Net / SVM / XGBoost / MLP / LSTM / Transformer` 复现共识标签；其中 final signature 仍只用 `Elastic Net / SVM / XGBoost` 作为稳定筛选来源，避免让深度对照模型改写已定稿面板。
10. 将 latent-heavy signature 转译为 raw-marker signature，并收缩为当前 10 特征 final signature。

## 代码入口

- [src/stad_subtyping/io.py](src/stad_subtyping/io.py): 数据读取、样本 ID 处理和 QC。
- [src/stad_subtyping/preprocess.py](src/stad_subtyping/preprocess.py): 各模态预处理与降维。
- [src/stad_subtyping/snf.py](src/stad_subtyping/snf.py): 相似网络构建与 SNF。
- [src/stad_subtyping/cluster.py](src/stad_subtyping/cluster.py): 共识聚类、谱聚类和标签比较。
- [src/stad_subtyping/evaluate.py](src/stad_subtyping/evaluate.py): 临床、生存、Cox、图件和 subtype 解释。
- [src/stad_subtyping/models.py](src/stad_subtyping/models.py): early integration 与 NMF 对照模型。
- [src/stad_subtyping/consensus.py](src/stad_subtyping/consensus.py): 多模型共识和 copy-rescued 权重逻辑。
- [src/stad_subtyping/supervised.py](src/stad_subtyping/supervised.py): 监督分类器和交叉验证。
- [src/stad_subtyping/signature.py](src/stad_subtyping/signature.py): signature 证据聚合、raw-marker 转译、final panel 和解释层。
- [src/stad_subtyping/pipeline.py](src/stad_subtyping/pipeline.py): 完整流程编排。

## 提交提醒

课程 PDF 要求最终提交 `PPT + WORD`。当前 Word 已生成，PPT 尚需制作。

提交前请确认：

- Word 报告已补真实成员信息表。
- PPT 首页包含课题名称、组长姓名、抽签序号和成员人数。
- 文件命名采用 `（抽签序号）-课题组长姓名-成员人数`。
- 邮件提交时间为第 13 周前，即 5 月 31 日之前。
- 提交邮箱为 `bioinfordriver@163.com`。
