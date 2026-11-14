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
| P6 | `run_all.py` 一键复现、`04_repro.ipynb`、运行日志与报告统稿 | 完成 |
| 改进实验 | 10 折稳健性、概率校准、句向量对照、规则增强、聚类对比、边界复核 | 完成 |

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
10 折稳健性、校准与阈值见 `results/tables/classify_robustness.json`；句向量对照见 `results/tables/classify_embedding.json`；聚类方法对比见 `results/tables/cluster_compare.json`。
边界复核见 [docs/scope_audit.md](docs/scope_audit.md)，规则复核见 [docs/rules_review.md](docs/rules_review.md)。

## 已产出结果

- 解析后评论 2,565,349 行、商品 71,911 行；游戏本体 40,259 件商品、1,456,092 条评论，正向率 74.75%。
- 宽表 47 列，其中 36 个建模特征、31 个衍生特征；按 `asin` 分组做 5 折，同一商品不跨折。
- P3 关联规则：19,032 个购物篮；Apriori 与 FP-Growth 均得到 1,855 条规则、去重后 932 条，默认参数下 Apriori 77.9 秒、FP-Growth 1.2 秒；三组阈值分别得到 0、21、932 条规则，`support=0.02, confidence=0.15` 过高导致无规则；平凡规则去除 0 条、lift≤1 过滤 0 条；932 条中 34 条前件与后件同系列，conviction 中位数 3.21。
- P4 分类最佳模型为线性 SVM（SGD）：5 折宏 F1 0.8635、PR-AUC 0.9837、ROC-AUC 0.9569；与 Dummy 的 Wilcoxon 检验 `p=0.0625`，在 5 折下不显著。10 折复核宏 F1 0.8642±0.0088，配对差 0.4365（95% CI 0.4314–0.4417），`p=0.001953`，说明 5 折的 p 值主要是样本量下限。
- 概率校准：逻辑回归原始 ECE 0.088，分组 isotonic 校准后 0.00036，Brier 0.0849→0.0695；F1 最优阈值 0.4545，正类 F1 0.9374；代价敏感（FN=5×FP）阈值 0.17。
- 文本表示对照（30,000 条抽样、300 字符、10 折）：哈希 TF-IDF 宏 F1 0.7618，BGE-small 0.8298，配对差 0.0680（95% CI 0.0507–0.0856），`p=0.001953`；方向与效应量支持句向量，但仍只是短文本抽样，不能外推全量。
- P5 聚类：40,259 件游戏本体商品、13 个聚类特征，轮廓系数选 `k=4`，bootstrap ARI 均值 0.9849（95% 区间 0.9296–0.9976）；`cluster_profiles.csv` 含每簇价格中位数、观测价格数、价格缺失率，其中一簇仅 1 件商品有观测价格，价格中位数不可单独解读。
- 聚类方法对比：GMM 轮廓 0.1441、HDBSCAN 2 簇轮廓 0.3119（但只有 Software 与其余两类）、有价子集 KMeans 轮廓 0.2102；KMeans k=4 仍适合运营分层。
- 边界复核：226 件强硬件词候选、7,105 条评论；前 30 条中 21 件纯硬件/主机、8 件游戏+配件同捆、1 件纯游戏，不能按标题一刀切；Retro 另有 1,154 件因缺 Games 节点被排除。
- 已有中文标注图 41 张：fig01–fig10 为 EDA，fig11–fig16 为规则，fig20–fig26 为聚类，fig30–fig35 为分类，fig36–fig39 为改进实验，fig40–fig42 为补充分析，fig43–fig46 为时间/否定/网络/留出验证。
- 时间外推（300,000 条抽样）：≤2014→2015+ 宏 F1 0.9048、PR-AUC 0.9934；2015+→≤2014 宏 F1 0.8249、PR-AUC 0.9738。2015 后逐年 0.9028–0.9133，没有衰减；用后期文本回测早期明显更难。
- 否定范围特征（150,000 条抽样）：宏 F1 0.8689→0.8785，否定子集错误率 15.95%→14.39%，转折 17.64%→16.35%；有效但未根治转折。
- 三张网络：also_buy 1,207,676 边、also_view 1,091,164 边、共同评论人 45,951 边；跨主品类占比 1.15%/1.77%/2.59%；also_buy 与 also_view 的 Jaccard 0.2539，与共同评论人仅 0.0102，说明推荐位网络比浏览/评论网络更窄。
- 聚类留出验证：按 asin 70/30 切分，12,078 件留出商品；评论量 ε²=0.437、also_buy ε²=0.479、评分仅 0.006，四簇仍按热度/共购分层而非评分高低。
- 静态业务看板：`results/dashboard.html`，含关键指标、四个运营分层、留出验证表与两张对比图。
- P6 七步全量已跑通，总耗时 847.3 秒，状态 `ok`；各步为 parse 93.4、scope 4.7、build_features 109.9、governance 15.3、rules 101.5、classify 492.5、cluster 29.6 秒。该数字是 P6 合入时的基线；后续改进只单独重跑了 rules 与 cluster，未重跑整链。

## 改进实验

这组实验不改变主流水线的样本口径与七步结果，只用于回答“结论是否稳、还能怎么改进”：

- 10 折分组复验：线性 SVM 宏 F1 0.8642±0.0088，与 Dummy 的配对差 0.4365（95% CI 0.4314–0.4417），`p=0.001953`；5 折的 0.0625 主要是样本量下限，不是“无差异”。
- 概率校准：逻辑回归原始 ECE 0.0881，分组 isotonic 校准后 0.00036；Brier 0.0849→0.0695。校准后 F1 最优阈值 0.4545，代价敏感（FN=5×FP）阈值 0.17。
- 文本表示对照：30,000 条抽样、300 字符、10 折，BGE-small 宏 F1 0.8298，哈希 TF-IDF 0.7618；配对差 0.0680（95% CI 0.0507–0.0856），`p=0.001953`。方向支持句向量，但仍只是短文本抽样，不能外推到全量。
- 规则增强：给 932 条规则增加 conviction 与同系列标记；conviction 中位数 3.21，34 条同系列，top 规则仍集中在 hidden-object 下载版互推。`also_buy` 仍只是推荐位，不是真实订单。
- 聚类对比：GMM 轮廓 0.1441；HDBSCAN 只分 2 簇（轮廓 0.3119）；有价子集 KMeans k=4 轮廓 0.2102；均未优于 KMeans k=4 的四段运营分层。
- 边界复核：226 件强硬件词候选（7,105 条评论），前 30 条中 21 件纯硬件/主机、8 件同捆、1 件纯游戏；主口径不按标题一刀切，Retro 另有 1,154 件缺 Games 节点。详见 [docs/scope_audit.md](docs/scope_audit.md)。
- 时间外推：按评论年份做正反两向切分，正向 2015–2018 宏 F1 0.9028–0.9133，稳定；反向早期 2003–2006 降到 0.756–0.776，说明后期文本不能直接回测早期评论。
- 否定专项：否定范围特征把宏 F1 从 0.8689 提到 0.8785，否定/转折子集错误率分别下降 1.56/1.29 个百分点；仍是“小增益”，不是根治。
- 三张网络：also_buy 与 also_view 的边重叠 Jaccard 0.2539，与共同评论人仅 0.0102；跨主品类边占比 1.15%/1.77%/2.59%，都不能支持“跨品类购买”结论。
- 聚类外部验证：训练集拟合、留出集按最近质心分配后，评论量/also_buy 的 ε² 为 0.437/0.479，评分仅 0.006，簇的运营含义仍是热度与共购，而非评分。
- 业务看板：`results/dashboard.html` 把四个分层、留出验证、时间/否定/网络结论放到一页；静态 HTML，无服务依赖。

这些脚本默认不写入 `run_all.py` 的七步主链，避免把可选的解释性实验变成每次全量重跑的成本；单独命令见“运行顺序”。

## 核心发现

- 评分的驱动力几乎全在文本：仅文本宏 F1 0.8727，合并元数据反而 0.8635，仅元数据 0.5788。元数据不是评分主特征，更适合解释分群。
- 词级系数把“为什么”落到具体词：great/best/awesome/love 推高评分，not/worst/boring/not worth 拉低评分；含否定词的样本错误率 12.48%，含转折词 13.95%，否定与转折仍是主要错误来源。
- 四个簇分开的是热度和共购结构，不是评分高低：簇 1 高热度、共购密集（平均评分 4.011），适合系列捆绑；簇 0 评分 3.641、verified 0.700，适合质量排查；簇 2 是 45.8% 的长尾，平均 5.9 条评论，适合搜索/自动化。
- 关联规则的头部是同系列/同平台组合；跨系列 support 中位数 0.0059，没有稳定跨品类组合。`also_buy` 只能当推荐位网络，不是真实购买网络。
- 时间维度上，2015 年后的文本驱动是稳定的；但用后期文本回测早期会明显变难，早期评论不能当同分布样本。
- 否定范围特征有小幅稳定增益；真正的错误来源仍是转折、反讽和短文本，继续加词表不如换句法/句向量。
- 四簇在留出集仍按热度与共购结构分开，评分差异很弱；因此簇名应写成“高热度/长尾/Software”，不能写成“高评分/低评分”。

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
│   ├── classify_robustness.py  10 折复核、校准与业务阈值
│   ├── classify_embedding.py   哈希 TF-IDF 与 BGE 句向量对照
│   ├── classify_ablation.py    文本 / 元数据 / 合并的模态消融
│   ├── classify_terms.py       词级系数与否定/转折错误率
│   ├── rules.py            also_buy 共购关联规则与阈值对比
│   ├── rules_linkage.py    规则同系列 / 跨系列与品类联动
│   ├── cluster.py          商品聚类、簇画像与稳定性
│   ├── cluster_compare.py  KMeans / GMM / HDBSCAN / 有价子集对比
│   ├── cluster_insights.py  簇与评分/评论量/verified/价格观测联动
│   ├── cluster_validate.py  按 asin 留出集验证簇的预测性
│   ├── classify_time.py     按评论年份做时间外推回测
│   ├── classify_negation.py 否定范围特征与否定/转折错误率
│   ├── network_compare.py   also_buy / also_view / 共同评论人三网对比
│   ├── build_dashboard.py   生成 results/dashboard.html 静态看板
│   ├── scope_audit.py      强硬件词与 Retro 边界复核清单
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
├── requirements-optional.txt  句向量对照的可选依赖
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

10 折稳健性、效应量、校准与业务阈值：

```
python src/classify_robustness.py
```

句向量对照（可选，需先安装 `fastembed`，首次会下载 BGE 模型）：

```
pip install -r requirements-optional.txt
set HF_ENDPOINT=https://hf-mirror.com
python src/classify_embedding.py --limit 30000 --text-chars 300 --folds 10
```

聚类方法对比、簇业务联动与边界复核：

```
python src/cluster_compare.py
python src/cluster_insights.py
python src/scope_audit.py
```

模态消融、词级解释与规则联动：

```
python src/classify_ablation.py
python src/classify_terms.py
python src/rules_linkage.py
```

时间外推、否定专项、三张网络、留出验证与业务看板：

```
python src/classify_time.py --limit 300000
python src/classify_negation.py --limit 150000
python src/network_compare.py
python src/cluster_validate.py
python src/build_dashboard.py
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
