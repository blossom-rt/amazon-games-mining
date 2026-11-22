# 文档索引

本目录保存项目的数据说明、方法报告与运行指南。英文入口见 [docs/en](en/README.md)，中文文档是详细分析的主版本。

## 数据与口径

| 文档 | 内容 |
|---|---|
| [data_source](data_source.md) | 数据来源、文件校验、字段与引用 |
| [data_dictionary](data_dictionary.md) | 原始字段、缺失率与清洗规则 |
| [feature_dictionary](feature_dictionary.md) | 47 列宽表与 36 个建模特征 |
| [scope_definition](scope_definition.md) | 游戏本体边界与 in_scope 规则 |
| [scope_audit](scope_audit.md) | 边界候选与 Retro 复核清单 |

## 方法与结果

| 文档 | 内容 |
|---|---|
| [governance_report](governance_report.md) | 缺失机制、填补、异常值与一致性检查 |
| [classification_report](classification_report.md) | 分类模型、5/10 折、校准、错误分析与 SHAP |
| [rules_review](rules_review.md) | 关联规则、阈值敏感性、同系列与冗余检查 |

## 运行与展示

| 文档 | 内容 |
|---|---|
| [setup](setup.md) | 环境、依赖、字体与常见问题 |
| [figure_reproduction](figure_reproduction.md) | 44 张图的重绘、输入依赖与校验 |
| [dashboard_guide](dashboard_guide.md) | Streamlit 看板、联动筛选与公网入口 |
| [streamlit_deploy](streamlit_deploy.md) | Streamlit Community Cloud 部署 |

英文完整入口：[docs/en](en/README.md)，包含 setup、数据、治理、分类、规则、聚类、看板与图复现文档。
