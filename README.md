# Amazon 游戏商品数据的探索与挖掘

基于 Amazon Review Data (2018) 的 Video_Games 品类，融合两类异构数据：

- **用户评论**：非结构化文本 + 评分、时间、verified 等字段。
- **商品元数据**：半结构化 JSON，含类目、品牌、价格、`also_buy`、`also_view` 等。

项目覆盖数据治理、关联规则、评论分类、商品聚类、可解释性与一键复现。

## 快速导航

| 文档主题 | 入口 |
|---|---|
| 数据来源、字段与边界 | [data_source](docs/data_source.md)、[data_dictionary](docs/data_dictionary.md)、[scope_definition](docs/scope_definition.md) |
| 特征与折划分 | [feature_dictionary](docs/feature_dictionary.md) |
| 治理与缺失/异常 | [governance_report](docs/governance_report.md)、[scope_audit](docs/scope_audit.md) |
| 分类模型与错误分析 | [classification_report](docs/classification_report.md) |
| 规则复核 | [rules_review](docs/rules_review.md) |
| 环境与看板 | [setup](docs/setup.md)、[dashboard_guide](docs/dashboard_guide.md)、[streamlit_deploy](docs/streamlit_deploy.md) |
| 项目结构 | [目录结构](#目录结构) |

## 当前进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | gzip JSON Lines 解析为 parquet，字段字典回填 | 完成 |
| P1 | 缺失机制、填补策略、异常值、一致性检查与 EDA | 完成 |
| P2 | 评论 × 商品宽表，时间窗与交互特征，分层分组折 | 完成 |
| P2.5 | 游戏本体边界判定，固定 P3/P4/P5 样本口径 | 完成 |
| P3 | `also_buy` 关联规则，Apriori / FP-Growth 对比 | 完成 |
| P4 | 评论情感分类，5 折与 10 折评估，错误分析与 SHAP | 完成 |
| P5 | 商品聚类，肘部法、轮廓系数、bootstrap ARI | 完成 |
| P6 | `run_all.py`、`04_repro.ipynb`、运行日志与结果归档 | 完成 |
| 补充分析 | 校准、句向量、消融、词级、时间、网络、链接预测、决策曲线 | 完成 |
| 交互看板 | Streamlit 4 视图、2 个联动筛选 | 完成 |

P3、P5 由[协作成员](https://github.com/coldcodefish)完成；P6 与看板由[仓库维护者](https://github.com/blossom-rt)负责。`main` 只保留稳定版本，稳定后由 `dev` 合并。

## 数据概览

| 项目 | 数值 |
|---|---|
| 评论 | 2,565,349 条 |
| 商品原始记录 | 84,819 条（官方标称 84,893） |
| 商品去重后 | 71,911 件 |
| 游戏本体样本 | 40,259 件商品 / 1,456,092 条评论 |
| 正向评论（4–5 星） | 74.75% |
| 宽表 | 2,565,349 × 47；36 个建模特征、31 个衍生特征 |
| 时间范围 | 1997-11 — 2018-10 |

两个源文件放在 `data/raw/`，不纳入版本控制：

```text
Video_Games.json.gz        493.7 MB   评论
meta_Video_Games.json.gz    50.8 MB   商品元数据
```

校验值、官方字段与引用格式见 [data_source](docs/data_source.md)。

## 关键结果

### 关联规则

19,032 个购物篮，`also_buy` 只保留游戏本体商品对。

| support / confidence | 原始规则 | 去重后 | Apriori | FP-Growth |
|---|---:|---:|---:|---:|
| 0.020 / 0.15 | 0 | 0 | 0.00 s | 0.04 s |
| 0.010 / 0.20 | 42 | 21 | 0.34 s | 0.09 s |
| 0.005 / 0.10 | 1,864 | 932 | 72.76 s | 9.09 s |
| 默认（0.005 / 0.20） | 1,855 | 932 | 66.5 s | 8.4 s |

- FP-Growth 在低支持度组约快 **8 倍**，规则集与 Apriori 完全一致。
- 932 条规则中 34 条为同系列，conviction 中位数 3.21。
- 头部规则集中在同系列 / hidden-object 下载版互推，跨系列没有稳定组合。

### 评论分类

5 折 OOF，按 `asin` 分组，同一商品不跨折。

| 模型 | 宏 F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|
| Dummy | 0.4277 | 0.7475 | 0.5000 |
| 逻辑回归（SGD） | 0.8583 | 0.9838 | 0.9551 |
| 线性 SVM（SGD） | **0.8635** | **0.9837** | **0.9569** |
| HistGradientBoosting | 0.7849 | 0.9701 | 0.9160 |
| 随机森林 | 0.7747 | 0.9673 | 0.9086 |

- 5 折 Wilcoxon `p=0.0625`；10 折复验 `p=0.001953`，说明 5 折的 p 值主要是样本量下限。
- 仅文本宏 F1 **0.8727**，合并元数据反而降到 **0.8635**；仅元数据只有 0.5788。
- BGE-small 全量 1,456,092 条、10 折、300 字符、RTX 4060 GPU：宏 F1 **0.8658**，哈希 TF-IDF 0.8197；配对差 0.0461，95% CI [0.0419, 0.0506]，`p=0.001953`；嵌入 1,052 秒、全流程 2,580 秒。60k 抽样为 0.8243 对 0.7540，方向一致。
- 口径注意：主分类用 3,000 字符哈希 TF-IDF，宏 F1 0.8635；全量 BGE 只高 0.0023，且嵌入成本高，因此 BGE 是短文本/句义场景的补充，不替换主模型。
- 否定范围特征把宏 F1 从 0.8689 提到 0.8785；否定子集错误率 15.95%→14.39%，转折 17.64%→16.35%。

### 商品聚类

40,259 件游戏本体商品、13 个聚类特征，`k=4`：

- 轮廓系数 **0.209**
- bootstrap ARI **0.9849**（95% 区间 0.9296–0.9976）

| 簇 | 商品数 | 平均评分 | 平均评论量 | verified | 价格观测率 |
|---|---:|---:|---:|---:|---:|
| 0 中等热度·评分偏低 | 7,797 | 3.641 | 25.4 | 0.700 | 0.0001 |
| 1 高热度·共购密集 | 11,877 | 4.011 | 94.5 | 0.638 | 0.1511 |
| 2 长尾·共购稀疏 | 18,430 | 3.747 | 5.9 | 0.662 | 0.1814 |
| 3 中等热度·Software | 2,155 | 3.671 | 12.5 | 0.709 | 0.3926 |

四簇主要按热度与共购结构分开，评分差异很弱；簇名宜按热度/共购结构解释，不宜表述为评分分层。

### 工程复现

七步全量 847.3 秒，状态 `ok`：

| 步骤 | 耗时 |
|---|---:|
| parse | 93.4 s |
| scope | 4.7 s |
| build_features | 109.9 s |
| governance | 15.3 s |
| rules | 101.5 s |
| classify | 492.5 s |
| cluster | 29.6 s |

运行日志写入 `results/tables/run_log.json`，含 git 提交、耗时、硬件与每步产物。

## 补充实验

补充实验不改变七步主链的样本口径，用于检验稳健性与业务解释。

### 评估稳健性

| 实验 | 结果 | 结论 |
|---|---|---|
| 10 折复验 | SVM 宏 F1 0.8642±0.0088；配对差 0.4365（95% CI 0.4314–0.4417） | 5 折 p=0.0625 不是“无差异” |
| 概率校准 | ECE 0.088→0.00036；Brier 0.0849→0.0695 | isotonic 明显改善概率质量 |
| 句向量 | 全量 300 字符：BGE 0.8658 vs TF-IDF 0.8197；p=0.001953 | 短文本有增益，但主模型 0.8635 仅差 0.0023，成本高 |
| 时间外推 | 2015+ 宏 F1 0.9028–0.9133；回测 2003–2006 降到 0.756–0.776 | 2015 后稳定，早期样本不宜视为同分布 |
| 模态消融 | 文本 0.8727 / 元数据 0.5788 / 合并 0.8635 | 文本是主信号，元数据没有稳定增量 |

### 业务解释与网络结构

| 实验 | 结果 | 结论 |
|---|---|---|
| 聚类外部验证 | 留出 12,078 件；评论量 ε²=0.437、also_buy ε²=0.479、评分仅 0.006 | 簇仍按热度/共购分层 |
| 聚类方法对比 | GMM 0.1441；HDBSCAN 2 簇 0.3119；有价子集 0.2102 | 均未优于 KMeans k=4 |
| 三张网络 | also_buy / also_view / 共同评论人；跨主品类 1.15% / 1.77% / 2.59% | 推荐网络比浏览/评论网络更窄 |
| 链接预测 | 共同邻居 Recall@10 0.436、Recall@50 0.725；热度 0.010 | 推荐位可由共现结构补全，但不是购买预测 |
| 边界复核 | 226 件强硬件词候选；前 30 条中 21 纯硬件、8 同捆、1 纯游戏 | 边界判定不按标题一刀切；Retro 另有 1,154 件被排除 |

### 业务阈值

20 万抽样、按 `asin` 5 折，isotonic 校准：

| 用途 | 阈值 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 人工审核起点（FN=5×FP） | 0.17 | 0.871 | 0.987 | 0.925 |
| 高置信自动通过 | 0.46 | 0.928 | 0.955 | 0.941 |
| 默认阈值 | 0.50 | 0.934 | 0.948 | 0.941 |

- 0.17 的净收益 0.714，全部判正为 0.694。
- 0.46 的净收益 0.666，全部判正为 0.530。
- 0.01 的净收益最高，但已接近“全部判正”，不适合当运营阈值。

## 核心发现

1. **评分的驱动力几乎全在文本。** 仅文本宏 F1 0.8727，高于合并元数据的 0.8635；元数据更适合解释分群，不是评分主特征。
2. **否定和转折是主要错误来源。** 含否定词样本错误率 12.48%，转折 13.95%；否定范围特征只有小增益，转折问题尚未根治。
3. **四个商品簇是运营分层，不是评分分层。** 簇 1 高热度、共购密集；簇 2 是 45.8% 的长尾；评分差异 ε² 仅 0.006。
4. **`also_buy` 是推荐位网络，不是购买网络。** 头部规则是同系列 / 同平台；跨系列 support 中位数 0.0059，没有稳定跨品类组合。
5. **推荐位补全可预测，但不等同于购买预测。** 共同邻居 Recall@10 0.436，热度只有 0.010，`also_view` 余弦 0.266。
6. **分类概率适合“阈值 + 人工审核”。** 0.46 可作高置信自动通过，0.17 可作审核起点；低阈值净收益被 74.6% 的正类率主导。

## 合规补充实验

对应课程表 1 的硬指标与可选项，补充了以下对照：

| 项 | 结果 | 结论 |
|---|---|---|
| 嵌套调参 | 外层 5 折 × 内层 3 折 RandomizedSearch；默认/调参宏 F1 0.8631/0.8631 | 固定参数已接近平台，调参流程有效但无提升 |
| 分群误差 | 总体 8.87%；3 星 37.24%、4 星 13.66%、5 星 2.67% | 3 星中性和早期语料最难，需优先人工复核 |
| 不平衡处理 | 原始 / class_weight / 欠采样：LR 0.8778/0.8683/0.8600，SVM 0.8821/0.8702/0.8644 | 重加权与欠采样都降低宏 F1，保留原始分布 + 阈值 |
| 特征选择 | 34 → 32 → top15；数值特征 F1 0.6205 vs 0.6203 | 无提升；PCA 前 10 主成分 58.15%，t-SNE 正负类重叠 |
| 规则冗余与内存 | Apriori 72.76 s / 15.5 GB；FP-Growth 9.09 s / 27.5 MB；平凡/冗余规则 0 | FP-Growth 更适合复现，Apriori 接近内存上限 |

数据字典已补字段级单位、来源与时点；表 1 的“或”项（Notebook 与看板、PCA/t-SNE 与特征选择）都保留了两套实现。

## 交互看板

`src/dashboard_app.py` 使用 Streamlit + Plotly，包含 4 个视图：

- **总览**：样本量、评分、评论量、时间外推 F1。
- **分群**：四个运营簇的规模与业务指标。
- **时间**：逐年宏 F1 与 PR-AUC。
- **网络**：三张网络的跨品类比例与边重叠。

侧边栏的两个筛选器：

- **运营簇**：联动总览 KPI 和分群视图。
- **评论年份**：联动总览时间 F1 和时间曲线。

启动：

```bash
pip install -r requirements-dashboard.txt
streamlit run src/dashboard_app.py --server.port 8501
```

访问 `http://127.0.0.1:8501`；健康检查 `/_stcore/health`。

公网访问地址：

- **静态交互版（推荐，无需服务，本次提交合入 `dev` 后生效）**：`https://raw.githack.com/blossom-rt/amazon-games-mining/dev/results/dashboard.html`。支持 `?tab=overview|clusters|time|network`，含簇多选与年份滑块两个联动筛选。
- **Streamlit 临时演示**：`https://quantitative-horses-fastest-close.trycloudflare.com`。由 Cloudflare Quick Tunnel 转发到本机 8501，进程重启后会换地址；长期部署见 [streamlit_deploy](docs/streamlit_deploy.md)。

不启动服务时也可直接打开 `results/dashboard.html`。部署细节见 [dashboard_guide](docs/dashboard_guide.md)。

## 目录结构

```text
amazon-games-mining/
├── data/                 原始与处理后的数据，不入库
├── src/                  可执行脚本
│   ├── config.py
│   ├── parse.py / scope.py / build_features.py
│   ├── governance.py
│   ├── rules.py / rules_linkage.py
│   ├── classify.py / classify_robustness.py
│   ├── classify_embedding.py / classify_ablation.py
│   ├── classify_terms.py / classify_time.py
│   ├── classify_negation.py / classify_decision.py
│   ├── classify_tuning.py / classify_subgroups.py
│   ├── classify_imbalance.py / feature_selection.py
│   ├── cluster.py / cluster_compare.py
│   ├── cluster_insights.py / cluster_validate.py
│   ├── network_compare.py / link_prediction.py
│   ├── dashboard_app.py / build_static_dashboard.py
│   ├── scope_audit.py
│   ├── run_all.py
│   └── io_utils.py
├── notebooks/            01 EDA、02 规则、03 聚类、04 复现
├── results/              figures 与 tables
├── docs/                 数据、字段、治理、分类、看板文档
├── requirements.txt
├── requirements-optional.txt
├── requirements-dashboard.txt
└── README.md
```

## 环境

```bash
conda create -n amazon-games python=3.11 -y
conda activate amazon-games
pip install -r requirements.txt
python -m ipykernel install --user --name amazon-games --display-name "Python (amazon-games)"
```

若默认 pip 镜像不可用，加 `-i https://pypi.org/simple`。

图表中文需要系统中文字体；Windows 一般已有 `SimHei`，其他系统见 [setup](docs/setup.md)。

## 分支与协作

| 分支 | 用途 | 规则 |
|---|---|---|
| `main` | 稳定版本 | 只通过 PR 合入 |
| `dev` | 集成分支 | 日常开发基线 |
| `feat/*` | 短期功能分支 | 从 `dev` 开，PR 回 `dev` |

已合并的功能分支：

| 分支 | 内容 |
|---|---|
| `feat/rules` | P3 关联规则 |
| `feat/cluster` | P5 商品聚类 |
| `feat/repro` | P6 一键复现 |
| `feat/insights` | 消融、词级、时间、网络、看板 |

链接预测、决策曲线与合规补充实验已合入 `dev`（PR #14）。GPU/BGE、`run_all --extras`、CI、Streamlit 部署与静态看板增强目前在 `dev` 工作区，提交后 PR 回 `dev`。

开始新工作：

```bash
git switch dev
git pull --ff-only
git switch -c feat/<阶段名>
```

提交：

```bash
git add <本阶段文件>
git commit -m "feat(<阶段>): ..."
git push -u origin feat/<阶段名>
```

功能分支通过 PR 合入 `dev`；`data/`、`__pycache__`、`.ipynb_checkpoints` 不纳入版本控制。

## 运行命令

### 数据准备

```bash
python src/download.py --verify
python src/download.py
python src/parse.py
python src/scope.py
python src/build_features.py
```

### 主流水线

```bash
python src/governance.py
python src/rules.py
python src/classify.py
python src/cluster.py
```

### 补充实验

```bash
python src/classify_robustness.py
python src/classify_ablation.py
python src/classify_terms.py
python src/classify_time.py --limit 300000
python src/classify_negation.py --limit 150000
python src/classify_decision.py
python src/classify_tuning.py
python src/classify_subgroups.py
python src/classify_imbalance.py
python src/feature_selection.py
python src/network_compare.py
python src/link_prediction.py --sources 2000
python src/cluster_compare.py
python src/cluster_insights.py
python src/cluster_validate.py
python src/scope_audit.py
```

句向量对照可选，需安装额外依赖并首次下载模型：

```bash
pip install -r requirements-optional.txt
set HF_ENDPOINT=https://hf-mirror.com
python src/classify_embedding.py --full --text-chars 300 --folds 10 --backend sentence-transformers --device cuda
```

CPU 环境把 `--backend` 改回 `fastembed`、`--device cpu`；没有 GPU 时不要用 `sentence-transformers` 跑 6 万条。

### 一键复现

```bash
python src/run_all.py --check
python src/run_all.py --check --extras
python src/run_all.py --check --require-all
python src/run_all.py --extras
```

全量重跑约 14 分钟；`run_log.json` 会记录 git 提交、耗时、硬件与每步产物。`--extras` 额外执行链接预测、决策曲线、调参、分群误差、不平衡与特征选择，全部结果可在 `results/` 查看。

### CI

`.github/workflows/ci.yml` 在 push/PR 到 `dev`、`main` 时：
1. 缓存 `data/raw`，下载并校验两个 gzip 文件；
2. 运行 `parse → scope → build_features`；
3. 执行 `python src/run_all.py --check --require-all --only parse,scope,build_features`，只校验 CI 实际生成的三个步骤；分类/规则/聚类产物不在 CI 中训练。

因为 `data/processed/` 不入库，CI 用缓存与重新生成保证检查针对真实产物，而不是空文件。

## 复现约定

- 随机步骤统一使用 `SEED = 42`（`src/config.py`）。
- 在项目根目录运行；或使用 `python -m src.脚本名`。
- 中间产物写入 `data/processed/`，图表与表格写入 `results/`，均不入库。

## 已知限制

- 价格缺失 83.44%，四种填补的 R² 都为负；价格只做事后画像，不作为强特征。
- `also_buy` 是推荐位，不是真实购买篮；规则与链接预测只能解释推荐结构。
- 226 件强硬件词候选中，前 30 条里 8 件是同捆；主口径不按标题一刀切。
- BGE 对照已跑到全量，但文本截断 300 字符；主模型用 3,000 字符 TF-IDF，BGE 只高 0.0023，不能据此宣称句向量全面替代 TF-IDF。
- 聚类轮廓系数 0.209，簇间分离偏弱，更适合运营分层而非“自然类别”。
