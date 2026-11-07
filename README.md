# TCGA-STAD 多组学分型

生物信息学实验课作业：用 TCGA 胃腺癌（STAD）的五到六组学数据做整合分型，配套临床/生存评估、多模型共识、监督分类器复现和 compact signature 筛选。

结论与关键产出见 [docs/FINDINGS.md](docs/FINDINGS.md)。

## 快速开始

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m stad_subtyping pipeline \
  --data-dir biography_data \
  --out-dir results \
  --seed 2026
```

`src` 不在默认路径上时设一下 `PYTHONPATH=src`。旧入口 `scripts/run_full_pipeline.py` 参数完全一样，内部转发到同一个函数。

只补跑监督分类器（要求 `results/` 里已有完整 pipeline 的产物）：

```bash
.venv/bin/python -m stad_subtyping classifier \
  --results-dir results \
  --seed 2026 \
  --classifiers mlp lstm transformer
```

全量跑一次要几十分钟，改文档或报告时不用重跑。

## 命令

| 命令 | 输入 | 产出 |
| --- | --- | --- |
| `pipeline` | `biography_data/` 原始多组学表 | `results/` 下的主分析、扩展验证、对照模型、共识、分类器、signature |
| `classifier` | `results/classifiers/elastic_net/*` 缓存 | 指定分类器目录 + 更新 `results/classifier_summary.tsv` |
| `scripts/export_report_docx.py` | `docs/STAD多组学分型实验报告.md` | 同名 `.docx` 正式报告 |

## 配置

数据目录、模态清单、清洗参数都在 `src/stad_subtyping/config.py`。

| 配置 | 含义 |
| --- | --- |
| `MODALITIES` | 每组学的文件名候选、特征列、离散/连续、缺失率与 MAD 筛选阈值 |
| `ANALYSIS_QUEUES` | 分析队列：主发现队列与扩展验证队列各用哪几组学、产出到哪个子目录 |
| `--seed` | 全流程随机种子，默认 2026 |

## 项目结构

```
src/stad_subtyping/
├── config.py      数据清单、清洗参数、分析队列
├── io.py          读表、探针注释、缺失填充
├── features.py    特征筛选与组装
├── cluster.py     SNF 聚类与评估
├── consensus.py   多模型共识
├── evaluate.py    临床与生存分析
├── supervised.py  分类器训练与导出
├── signature.py   compact signature 筛选
├── pipeline.py    编排
├── extension.py   在已有缓存上补跑分类器
└── cli.py         子命令
scripts/           报告导出与两个转发脚本
docs/              结论文档、正式报告
tests/             28 个 pytest 用例 + 9 个不依赖 numpy 的 CLI 用例
```

## 测试

```bash
.venv/bin/pytest -q                    # 全量，需要 numpy/pandas 等
python -m unittest tests.test_cli      # 只测参数解析与包入口，不需要科学计算依赖
```

## 已知问题

- 全量 pipeline 是单进程顺序跑，没有并行；换机器跑之前先确认内存够放五组学矩阵。
- `lstm` / `transformer` 是 tabular-sequence 对照模型，把表格特征重排成伪序列，不代表真实时序结构。
- 报告导出脚本硬编码了 `docs/` 下的 markdown 路径，换文件名要改代码。
- `results/` 下的产物随仓库提交，历史运行记录靠文件命名区分，没有版本化数据集。

## License

见 [LICENSE](LICENSE)。
