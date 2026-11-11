# Amazon 游戏商品数据的探索与挖掘

## 项目简介

基于 Amazon Review Data (2018) 的 Video_Games 品类，融合用户评论（非结构化文本）与商品元数据（半结构化 JSON）两类异构数据源，完成一条完整的数据挖掘流程：数据治理、商品共购关联规则挖掘、评论情感分类、商品分群与结果可解释性分析。

数据来源与引用见 [docs/data_source.md](docs/data_source.md)。

## 数据

| 项目 | 内容 |
|---|---|
| 数据集 | Amazon Review Data (2018) |
| 发布方 | University of California, San Diego, McAuley Lab |
| 品类 | Video_Games |
| 商品数 | 84,819（官方标称 84,893） |
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
│   └── io_utils.py         流式读取 JSON Lines 的工具函数
├── notebooks/              探索性分析与出图
├── results/
│   ├── figures/            图表输出
│   └── tables/             表格输出
├── docs/                   数据说明、字段字典、治理报告与环境配置
├── requirements.txt
└── README.md
```

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

## 快速开始

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

构建评论 × 商品宽表前，先判定游戏本体边界：

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

训练评论情感分类模型，输出指标、曲线、错误样本与 SHAP：

```
python src/classify.py
```

## 复现约定

所有涉及随机性的步骤统一使用 `src/config.py` 中的 `SEED = 42`，并在文档中注明。
运行脚本前确保工作目录为项目根目录，或直接使用 `python -m src.脚本名`。
