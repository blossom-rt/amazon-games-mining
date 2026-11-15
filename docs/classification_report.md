# 评论情感分类报告（P4）

本报告对应表 1 的分类要求：基线 + 至少 3 个模型（含集成）、分层 5 折、均值±标准差、统计检验、≥3 个指标、混淆矩阵与 ROC/PR 曲线、错误样本解剖、全局重要性与 SHAP 局部解释。所有数字由 `python src/classify.py` 生成。

## 样本与标签

- 样本：`features.parquet` 中 `in_scope=True` 的游戏本体评论，共 1,456,092 条，正类率 74.75%。
- 标签：`label = 1` 当且仅当 `overall >= 4.0`；负类为 1–3 星。
- 折：直接读取 `folds.parquet` 的 `fold` 列，5 折来自 `StratifiedGroupKFold(groups=asin)`，同一商品的评论不跨折。
- 范围外评论（`fold = -1`）不进入训练、验证或指标计算。

## 防泄漏流程

每个训练折内依次拟合以下对象，验证折只做 `transform`：

1. 数值特征的中位数填补与缺失指示列（`SimpleImputer`）。
2. 数值特征标准化（`StandardScaler`）。
3. 类别特征的一位有效编码（`OneHotEncoder`，低频类别合并）。
4. 文本 TF-IDF 的 IDF 权重（`TfidfTransformer`）；词表由固定维度的 `HashingVectorizer` 提供，不使用验证折文本。
5. 树模型使用的 `TruncatedSVD` 只在训练折的 10 万行上拟合。

文本向量化在折外只做无监督的 hashing，不读取标签；IDF、SVD、填补、编码、标准化均不接触验证折。

## 文本与特征

- 文本：`reviewText + " " + summary`，截断到 3,000 字符。
- 向量化：`HashingVectorizer(n_features=262144, alternate_sign=False, norm=None)`，5 折共用同一个哈希空间，每折重新拟合 IDF。
- 线性模型输入：34 个数值特征 + 2 个类别特征的一位有效编码 + 262,144 维 TF-IDF，稀疏矩阵拼接。
- 树模型输入：34 个数值特征 + 20 维 SVD，避免在 26 万维稀疏矩阵上直接建树。
- 数值特征含时间窗 `reviews_prev_30d_same_asin` 与 `days_since_prev_review_same_asin`，口径见 `docs/feature_dictionary.md`。

## 模型

| 模型 | 关键设置 |
|---|---|
| Dummy | 按训练折先验预测，作基线 |
| 逻辑回归（SGD） | `loss=log_loss`，`alpha=1e-5`，`max_iter=8`，`class_weight=balanced` |
| 线性 SVM（SGD） | `loss=hinge`，`alpha=1e-5`，`max_iter=15`，`class_weight=balanced` |
| HistGradientBoosting | 120 轮，学习率 0.08，31 叶，早停，平衡样本权重 |
| 随机森林 | 60 棵，最大深度 18，叶最少 40，`balanced_subsample` |

类别不平衡通过 `class_weight`/样本权重处理；未做重采样，避免改变验证集分布。

## 结果

5 折均值±标准差，OOF 预测：

| 模型 | 准确率 | 宏 F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| Dummy | 0.7475 ± 0.0000 | 0.4277 ± 0.0000 | 0.7475 ± 0.0000 | 0.5000 ± 0.0000 |
| 逻辑回归（SGD） | 0.8849 ± 0.0100 | 0.8583 ± 0.0097 | 0.9838 ± 0.0011 | 0.9551 ± 0.0023 |
| 线性 SVM（SGD） | **0.8894 ± 0.0064** | **0.8635 ± 0.0065** | 0.9837 ± 0.0016 | **0.9569 ± 0.0024** |
| HistGradientBoosting | 0.8164 ± 0.0028 | 0.7849 ± 0.0027 | 0.9701 ± 0.0005 | 0.9160 ± 0.0016 |
| 随机森林 | 0.8066 ± 0.0033 | 0.7747 ± 0.0031 | 0.9673 ± 0.0005 | 0.9086 ± 0.0016 |

最佳模型为线性 SVM（SGD），宏 F1 0.8635；逻辑回归与其差距很小，说明主要信号来自 TF-IDF 词面。两个树模型明显更低，原因是 20 维 SVD 压缩了文本，且树模型不擅长 26 万维稀疏线性边界。

统计检验：最佳模型与 Dummy 的 5 折宏 F1 做 Wilcoxon 符号秩检验，`W = 0.0`，`p = 0.0625`；5 折样本量下这是最小可能 p 值，未达到 0.05 显著水平，正文按“方向一致但样本量不足”表述。

## 图

- `results/figures/fig30_classify_roc.png`：5 个模型的 OOF ROC 曲线。
- `results/figures/fig31_classify_pr.png`：5 个模型的 OOF PR 曲线。
- `results/figures/fig32_classify_confusion.png`：5 个模型的混淆矩阵。
- `results/figures/fig33_classify_importance.png`：HistGradientBoosting 的置换重要性前 20。
- `results/figures/fig34_classify_shap.png`：假正例、假负例、真正例各 1 个的 SHAP 局部贡献。
- `results/figures/fig35_classify_metrics.png`：宏 F1、PR-AUC、ROC-AUC 的均值±标准差。

## 错误样本解剖

`results/tables/classify_errors.csv` 取最佳模型 OOF 分数最高的 5 个假正例与最低的 5 个假负例，且同一商品只取一条。共同模式如下。

假正例（真实负向，预测正向）：

1. `B00IWMAREW`（Xbox 360 硬盘）：评论说“不能玩初代 Xbox 游戏，但除此之外很好”，负面事实被 `good` 压过。
2. `B01BHFI4XG`（ASUS VR 主机）：长文描述安装失败、蓝屏、最终退货，但开头有 `fun`，词面正向词占优。
3. `B000ZKA0J6`（游戏）：大段讨论游戏优点，只在结尾批评，整体词面偏正向。
4. `B006UIS53K`（Kinect 硬件）：评论称“paperweight”“no manual”，但 `Windows`、`games`、`useful` 等词与游戏评论同分布。
5. `B005OTUJXA`（Games 类目）：短评“not worth anything over 350$”，否定结构未被线性模型捕捉。

假负例（真实正向，预测负向）：

1. `B0009VXAM0`（PS3）：长文比较三台主机，含 `lower resolution`、`tighter budget` 等负面词，但总体是正向评价。
2. `B000I20TBU`：正文只有 `good` 四个字符，文本证据太少，模型被类别/商品特征拉向负向。
3. `B000EGELP0`（Brain Age）：评论先说 `hardly groundbreaking`、`monotonous`，后说 `strangely addictive`，转折句被前半段主导。
4. `B000CFMQPK`（Office Products / Games）：`easily installed and nice...` 是正向，但商品类目与游戏本体差异大。
5. `B000SL4ANE`（Chessmaster）：大量 `poor reviews`、`doesn't compare` 的转述，实际结论是推荐。

这些错误集中在四类：否定与转折、短文本、评分与文本不一致、以及范围判定仍混入的硬件/配件商品。最后一类与 `docs/scope_definition.md` 记录的 226 件强硬件词候选一致。

## 全局重要性与 SHAP

- 置换重要性前列几乎都是 SVD 维度（`svd_02`、`svd_13`、`svd_14` 等），其次是 `summary_len`、`review_word_count`、`vote_count`、`has_vote`；时间窗 `reviews_prev_30d_same_asin` 有少量贡献但远低于文本。
- SHAP 三个样本覆盖假正例、假负例、真正例：假正例由 `svd_02` 的正向贡献推高；假负例被 `svd_13` 与 `summary_len` 拉低；真正例同时受 `svd_02` 与 `summary_len` 影响。完整数值见 `results/tables/classify_shap.json`。

## 限制

- 线性 SVM 的决策分数不是校准概率；PR/ROC 不受影响，但业务阈值需要额外校准。
- 文本截断 3,000 字符，超长评论的后半段不进入模型。
- 否定、反讽、评分与文本冲突仍是主要错误来源；未使用句法或预训练语言模型。
- 范围判定保留的 226 件强硬件词候选会进入分类样本，错误样本中已出现硬件商品。
- 5 折 Wilcoxon 的 p 值下限为 0.0625，不能在本数据上宣称 0.05 显著；结论按方向与效应量表述。

## 复现

```bash
python src/classify.py
```

全量运行 779.7 秒，硬件为 Windows 10、28 核、15.7 GB 内存、Python 3.11.18。中间产物：

- `results/tables/classify_metrics.csv`：每折每模型指标。
- `results/tables/classify_summary.json`：均值±标准差、检验、耗时与硬件。
- `results/tables/classify_errors.csv`：10 条错误样本。
- `results/tables/classify_importance.csv`：置换重要性。
- `results/tables/classify_shap.json`：3 个局部样本的 SHAP 值。
- `results/tables/classify_run.log`：运行日志。
