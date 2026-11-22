# 图表复现

本仓库的 44 张结果图都可以从仓库内脚本和结果表重新生成，不依赖外部文档。

## 一键路径

全新 clone，先准备数据并运行主流水线与补充实验：

```bash
python src/download.py
python src/download.py --verify
python src/run_all.py --extras
```

`--extras` 的最后一步是 `figures`，会重绘可从结果表复现的图，并校验 44 张是否齐全。只想验收已有产物时：

```bash
python src/run_all.py --check --extras
python src/figure_pipeline.py --check
```

## 只重绘图

已有 `data/processed/`、`results/tables/` 时：

```bash
python src/figure_pipeline.py --all
python src/figure_pipeline.py --check
```

`--all` 会调用四个重绘模块，覆盖 33 张由结果表生成的图：

| 模块 | 覆盖图 |
|---|---|
| `redo_report_figures.py` | 01、08、13、38、40、41、42、44、46、47、48 |
| `redo_eda_rules_figs.py` | 03、04、05、06、09、12、14 |
| `redo_classify_figs.py` | 34、35 |
| `restyle_other_figures.py` | 02、07、10、11、15、16、22、24、26、36、37、39、45 |

另外 11 张由实验脚本直接写成最终样式，不从结果表重绘：

| 图 | 生成脚本 |
|---|---|
| 20、21、23、25 | `src/cluster.py` |
| 30、31、32、33 | `src/classify.py` |
| 43 | `src/classify_time.py` |
| 49 | `src/classify_subgroups.py` |
| 50 | `src/feature_selection.py` |

`figure_pipeline --all` 如果发现这 11 张缺失，会提示对应命令；`run_all --extras` 会先把它们生成出来，再执行 `figures` 校验。

## 输入依赖

| 输入 | 用途 | 生成方式 |
|---|---|---|
| `data/processed/reviews.parquet`、`meta.parquet`、`features.parquet`、`scope.parquet` | EDA、规则、校准、聚类图 | `parse → scope → build_features` |
| `results/tables/rules_*.csv` | fig11–16 | `rules.py` |
| `results/tables/cluster_assignments.parquet` | fig22、fig26 | `cluster.py` |
| `results/tables/cluster_profiles.csv`、`cluster_summary.json` | fig20–25 | `cluster.py` |
| `results/tables/classify_decision_oof.parquet`、`classify_decision_curve.csv`、`classify_decision.json` | fig36、37、48 | `classify_decision.py` |
| `results/tables/classify_embedding_metrics.csv` | fig38 | 已提交，或 `classify_embedding.py` |
| `results/tables/classify_subgroups.csv` | fig49 | `classify_subgroups.py` |
| `results/tables/feature_selection.json` | fig50 | `feature_selection.py` |
| `results/tables/classify_ablation_metrics.csv`、`classify_terms.csv`、`cluster_outcomes.csv`、`classify_negation_metrics.csv`、`network_compare.json`、`link_prediction_metrics.csv`、`classify_time_metrics.csv` | fig39–48 | 对应补充实验脚本，均已提交结果表 |

`data/processed/` 和 `*.parquet` 不入库，必须由数据准备步骤生成；`results/tables/` 与 `results/figures/` 随仓库提交，作为轻量复现输入。

结果表 JSON 中的 `outputs`/`source` 等字段统一写仓库相对路径；`run_all.py --extras` 与 `figure_pipeline.py --all` 结束时会调用 `sanitize_result_jsons()` 再清理一次，避免把本机绝对路径带进提交。

## 缺输入时

```bash
# 缺 processed 表
python src/run_all.py --only parse,scope,build_features

# 缺聚类分配或分类决策 OOF
python src/run_all.py --only cluster
python src/run_all.py --only classify_decision

# 只缺 fig43 / fig49 / fig50
python src/classify_time.py --limit 300000
python src/classify_subgroups.py
python src/feature_selection.py --limit 50000
```

`fig38` 默认读取已提交的 `classify_embedding_metrics.csv`，不会自动重跑 BGE；需要重跑时安装 `requirements-optional.txt`，命令见 README。

## 中文字体

图内文字依赖中文字体。Windows 通常自带 Microsoft YaHei；Linux 建议安装 `fonts-noto-cjk`：

```bash
# Debian/Ubuntu
sudo apt-get install fonts-noto-cjk
# macOS
brew install --cask font-noto-sans-cjk
```

脚本按 `Microsoft YaHei → SimHei → Noto Sans CJK SC → PingFang SC` 顺序查找。`figure_pipeline` 在缺字体时会打印警告；图中出现方框时先解决字体，不要改数据。

## 校验规则

`python src/figure_pipeline.py --check`：

- 固定检查 44 个文件名；
- 检查文件非空、可被 Pillow 打开、宽度至少 600 px、高度至少 300 px；
- 输出分组计数；任一缺失或损坏时退出码为 1；
- 不比较像素哈希，避免不同系统字体度量造成误报。

`python src/run_all.py --check --extras` 额外检查 `figures` 步骤的 44 张图数量。外部文档不参与校验。

CI 在每次 push/PR 时执行一次 `python src/figure_pipeline.py --check`，只验证已提交图片的完整性和可读性，不重绘图片；完整重绘仍在本地通过 `python src/run_all.py --extras` 完成。
