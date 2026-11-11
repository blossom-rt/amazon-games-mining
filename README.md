# Amazon 游戏商品数据的探索与挖掘

## 项目简介

基于 Amazon Review Data (2018) 的 Video_Games 品类，融合用户评论（非结构化文本）与商品元数据（半结构化 JSON）两类异构数据源，完成一条完整的数据挖掘流程：数据治理、商品共购关联规则挖掘、评论情感分类、商品分群与结果可解释性分析。

数据来源与引用见 [docs/data_source.md](docs/data_source.md)。

## 当前进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | gzip JSON Lines 解析为 parquet，字段字典回填 | 完成 |
| P1 | 缺失机制、填补策略、异常值、一致性检查与 EDA | 完成 |
| P2 | 评论 × 商品宽表、时间窗与交互特征、分层分组折 | 完成 |
| P2.5 | 游戏本体边界判定，固定 P3/P4/P5 的样本口径 | 完成 |
| P3 | `also_buy` 共购关联规则、Apriori 与 FP-Growth 对比 | 完成 |
| P4 | 评论情感分类、五折评估、错误分析与 SHAP | 完成 |
| P5 | 商品聚类、肘部法与轮廓系数、bootstrap ARI | 完成 |
| P6 | `run_all.py` 一键复现、`04_repro.ipynb`、运行日志与报告统稿 | 流水线完成，报告统稿待做 |

P0–P6 的代码与流水线已合入 `dev` 并推送；P3、P5 由[协作成员](https://github.com/coldcodefish)完成，P6 与报告统稿由[仓库维护者](https://github.com/blossom-rt)负责。`main` 只保留稳定版本，待报告定稿后由 `dev` 合并。

## 数据

| 项目 | 内容 |
|---|---|
| 数据集 | Amazon Review Data (2018) |
| 发布方 | University of California, San Diego, McAuley Lab |
| 品类 | Video_Games |
| 商品数 | 84,819（官方标称 84,893，去重后 71,911） |
| 评论数 | 2,565,349 |
| 时间范围 | 1997 年 11 月 — 2018 年 10 月 |
| 许可 | 仅限学术研究使用 |

两个源文件放在 `data/raw/`，不纳入版本控制：

```
data/raw/Video_Games.json.gz        493.7 MB   评论，一行一条 JSON
data/raw/meta_Video_Games.json.gz    50.8 MB   商品元数据，一行一件 JSON
```

文件校验值、官方字段说明与引用格式见 [docs/data_source.md](docs/data_source.md)。
各字段的类型、缺失率与取值范围见 [docs/data_dictionary.md](docs/data_dictionary.md)。
宽表字段、衍生特征与折划分口径见 [docs/feature_dictionary.md](docs/feature_dictionary.md)。
游戏本体边界规则与覆盖范围见 [docs/scope_definition.md](docs/scope_definition.md)。
一致性检查、缺失填补对比与异常值检测结果见 [docs/governance_report.md](docs/governance_report.md)。
评论情感分类的模型、指标与错误分析见 [docs/classification_report.md](docs/classification_report.md)。
关联规则的强度、阈值敏感性与运行时见 `results/tables/rules_*.csv/json`；聚类画像与 bootstrap 稳定性见 `results/tables/cluster_*.csv/json`。

## 已产出结果

- 解析后评论 2,565,349 行、商品 71,911 行；游戏本体 40,259 件商品、1,456,092 条评论，正向率 74.75%。
- 宽表 47 列，其中 36 个建模特征、31 个衍生特征；按 `asin` 分组做 5 折，同一商品不跨折。
- P3 关联规则：19,032 个购物篮；Apriori 与 FP-Growth 均得到 1,855 条规则、去重后 932 条，Apriori 96.35 秒、FP-Growth 1.15 秒；三组阈值分别得到 0、21、932 条规则，`support=0.02, confidence=0.15` 过高导致无规则；平凡规则去除 0 条、lift≤1 过滤 0 条。
- P4 分类最佳模型为线性 SVM（SGD）：宏 F1 0.8635、PR-AUC 0.9837、ROC-AUC 0.9569；与 Dummy 的 Wilcoxon 检验 `p=0.0625`，在 5 折下不显著，报告中按“优于基线但未达统计显著”表述。
- P5 聚类：40,259 件游戏本体商品、13 个聚类特征，轮廓系数选 `k=4`，bootstrap ARI 均值 0.9849（95% 区间 0.9296–0.9976）；`cluster_profiles.csv` 含每簇价格中位数、观测价格数、价格缺失率，其中一簇仅 1 件商品有观测价格，价格中位数不可单独解读。
- 已有中文标注图 29 张：fig01–fig10 为 EDA，fig11–fig16 为规则，fig20–fig26 为聚类，fig30–fig35 为分类。
- P6 七步全量已跑通，总耗时 847.3 秒，状态 `ok`；各步为 parse 93.4、scope 4.7、build_features 109.9、governance 15.3、rules 101.5、classify 492.5、cluster 29.6 秒。

## 目录结构

```
amazon-games-mining/
├── data/
│   ├── raw/                原始压缩文件，不入库
│   └── processed/          清洗后的 parquet，不入库
├── src/                    可执行脚本
│   ├── config.py           路径、常量、随机种子
│   ├── download.py         数据下载与完整性校验
│   ├── parse.py            gzip JSON Lines 转 parquet
│   ├── scope.py            判定商品是否为游戏本体
│   ├── build_features.py   评论 × 商品宽表与固定折
│   ├── governance.py       一致性检查、填补与异常值验证
│   ├── classify.py         评论情感分类、评估与解释
│   ├── rules.py            also_buy 共购关联规则与阈值对比
│   ├── cluster.py          商品聚类、簇画像与稳定性
│   ├── run_all.py          一键串联各步骤并写运行日志
│   └── io_utils.py         流式读取 JSON Lines 的工具函数
├── notebooks/              探索性分析与出图
│   ├── 01_eda.ipynb
│   ├── 02_rules.ipynb
│   ├── 03_cluster.ipynb
│   └── 04_repro.ipynb      复现入口与结果校验
├── results/
│   ├── figures/            图表输出
│   └── tables/             表格与运行日志
├── docs/                   数据说明、字段字典、治理与分类报告、环境配置
├── requirements.txt
└── README.md
```

`src/rules.py`、`src/cluster.py` 与对应 notebook 已在 P3、P5 合入；`src/run_all.py` 串起 parse → scope → build_features → governance → rules → classify → cluster 七步，`notebooks/04_repro.ipynb` 展示运行日志与关键指标。

## 环境

使用独立 conda 环境，Python 3.11，依赖版本已在 `requirements.txt` 中固定。

首次配置按 [docs/setup.md](docs/setup.md) 操作，其中覆盖环境创建、依赖安装、Jupyter 内核注册、数据下载与自检。最短路径：

```
conda create -n amazon-games python=3.11 -y
conda activate amazon-games
pip install -r requirements.txt
python -m ipykernel install --user --name amazon-games --display-name "Python (amazon-games)"
```

若默认的 pip 镜像不可用，安装时加 `-i https://pypi.org/simple`。图表中文标注需要系统中文字体，Windows 一般已带 `SimHei`，Linux 与 macOS 的处理见 setup 文档。

## 分支与协作

仓库只保留两条长期分支：

- `main`：稳定版本，只通过 Pull Request 合入，不直接提交。
- `dev`：集成分支，日常开发基线，P0–P6 已在此分支。

并行开发使用短期功能分支，完成后开 PR 合入 `dev`，合并后删除：

| 分支 | 内容 | 状态 |
|---|---|---|
| `feat/rules` | P3 关联规则、阈值敏感性、业务解读 | 已合并到 `dev` |
| `feat/cluster` | P5 商品聚类、簇画像、bootstrap ARI | 已合并到 `dev` |
| `feat/repro` | P6 一键复现、运行日志、复现 Notebook | 已合并到 `dev` |

开始新工作前先同步 `dev`：

```
git switch dev
git pull --ff-only
git switch -c feat/<阶段名>
```

提交并推送功能分支：

```
git add <本阶段文件>
git commit -m "feat(<阶段>): ..."
git push -u origin feat/<阶段名>
```

然后在 GitHub 开 PR，目标分支选 `dev`，不要直接推 `dev`，也不要自己合并。合并由仓库维护者完成。每天开始前执行 `git fetch origin` 与 `git rebase origin/dev`；只解决自己文件里的冲突。

不要提交 `data/` 下的 parquet、`__pycache__`、`.ipynb_checkpoints` 或任何原始数据。

## 运行顺序

检查本地数据是否完整（不下载、不解压）：

```
python src/download.py --verify
```

若文件缺失，重新下载：

```
python src/download.py
```

解析两个 gzip 文件，生成 parquet：

```
python src/parse.py
```

判定游戏本体边界：

```
python src/scope.py
```

构建评论 × 商品宽表，并给游戏本体固定分层分组折：

```
python src/build_features.py
```

复算一致性、比较缺失填补策略与异常值检测：

```
python src/governance.py
```

挖掘 `also_buy` 共购规则，比较 Apriori 与 FP-Growth，并输出阈值敏感性：

```
python src/rules.py
```

训练评论情感分类模型，输出指标、曲线、错误样本与 SHAP：

```
python src/classify.py
```

商品聚类、轮廓系数、bootstrap ARI 与簇画像：

```
python src/cluster.py
```

一键复现入口已可用。首次可先校验现有产物，不重跑：

```
python src/run_all.py --check
```

七步都必须存在时使用：

```
python src/run_all.py --check --require-all
```

全量重跑约 14 分钟：

```
python src/run_all.py
```

`run_all.py` 按 parse → scope → build_features → governance → rules → classify → cluster 的顺序串起整条流水线，并写入 `results/tables/run_log.json`。本机本次全量为 847.3 秒、状态 `ok`，其中分类 492.5 秒、规则 101.5 秒、聚类 29.6 秒。

## 复现约定

所有涉及随机性的步骤统一使用 `src/config.py` 中的 `SEED = 42`，并在文档中注明。
运行脚本前确保工作目录为项目根目录，或直接使用 `python -m src.脚本名`。
中间产物统一写入 `data/processed/`，图表与表格写入 `results/`，均不纳入版本控制。
