# 特征字典与宽表口径

## 产物

| 文件 | 规模 | 说明 |
|---|---|---|
| `data/processed/features.parquet` | 2,565,349 行 × 47 列，497.3 MB | 评论 × 商品宽表，一行一条评论，含 `in_scope` |
| `data/processed/folds.parquet` | 1,456,092 行 × 4 列，13.8 MB | 只含游戏本体，`reviewerID`、`asin`、`label`、`fold` |
| `results/tables/feature_summary.json` | — | 折分布、缺失率、耗时与硬件 |

构建命令：

```bash
python src/build_features.py --force
```

本次耗时 106.4 秒，环境为 Windows 10、28 核、15.7 GB 内存、Python 3.11.18、pyarrow 26.0.1。

## 连接口径

- 评论表 2,565,349 行左连接商品表 71,911 行，连接键 `asin`，校验 `many_to_one`。
- 商品表已按 `asin` 去重，原始 84,819 条记录中的 12,908 条完全重复记录不再进入宽表。
- 品类边界由 `src/scope.py` 判定：71,911 件商品中 40,259 件为游戏本体，覆盖 1,456,092 条评论（56.76%）；规则与排除原因见 `docs/scope_definition.md`。
- 宽表保留全部评论，`in_scope` 标记是否进入建模样本；`folds.parquet` 只写 `in_scope=True` 的行。
- 3,676 条评论的 `asin` 不在商品表，`has_meta=False`，`in_scope=False`，商品字段为缺失值或 `False`；这些评论没有删除。
- 宽表保留原始 `reviewText` 与 `summary`，文本清洗与向量化留到 P4，在训练折内拟合。

## 标签与折

- 标签定义：`label = 1` 当且仅当 `overall >= 4.0`；全量正类 1,899,779 条（74.06%），游戏本体正类率 74.75%。
- 折划分：`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`，分组键 `asin`，只在 `in_scope=True` 的 1,456,092 条评论上拟合。
- 折在任何编码、填补、标准化与特征选择之前生成，后续脚本只读取 `fold` 列，不允许重新划分。
- 同一 `asin` 的评论只落在一个折里，避免同一商品的评论同时出现在训练与验证。
- 范围外评论保留在宽表中，`fold = -1`，不进入建模与折统计。

| 折 | 行数 | 正类率 |
|---|---|---|
| 0 | 291,217 | 74.75% |
| 1 | 291,221 | 74.75% |
| 2 | 291,204 | 74.75% |
| 3 | 291,208 | 74.75% |
| 4 | 291,242 | 74.75% |

## 字段表

### 标识与目标

| 列名 | 类型 | 含义 | 缺失处理 |
|---|---|---|---|
| `reviewerID` | string | 评论人 ID | 无缺失 |
| `asin` | string | 商品 ID | 无缺失 |
| `unixReviewTime` | int64 | 评论时间戳，秒 | 无缺失 |
| `overall` | float32 | 原始评分 1–5，仅审计用 | 无缺失 |
| `label` | int8 | 目标变量，`overall >= 4` 为 1 | 无缺失 |
| `fold` | int8 | 分层分组折号 0–4；范围外为 -1 | 无缺失 |
| `in_scope` | bool | 是否属于游戏本体，由 `src/scope.py` 判定 | 无缺失 |

### 评论文本

| 列名 | 类型 | 含义 | 缺失处理 |
|---|---|---|---|
| `reviewText` | string | 原始评论正文 | 1,720 条缺失，保留空值 |
| `summary` | string | 原始评论标题 | 822 条缺失，保留空值 |

### 评论侧特征

| 列名 | 类型 | 含义 | 缺失处理 |
|---|---|---|---|
| `verified` | bool | 是否已验证购买 | 无缺失 |
| `review_year` | int16 | 评论年份，1997–2018 | 无缺失 |
| `review_month` | int8 | 评论月份 1–12 | 无缺失 |
| `review_dow` | int8 | 星期几，0 为周一 | 无缺失 |
| `review_len` | int32 | 正文字符数 | 空文本记 0 |
| `review_word_count` | int32 | 按空白切分的词数 | 空文本记 0 |
| `review_sentence_count` | int32 | 句末标点 `[.!?]` 段数加 1 | 空文本记 0 |
| `summary_len` | int32 | 标题字符数 | 空标题记 0 |
| `has_style` | bool | 是否有 `style` 规格信息 | 无缺失 |
| `has_vote` | bool | 是否有有用票字段 | 无缺失 |
| `vote_count` | int32 | 有用票数 | 缺失记 0；注意 0/1 票在原数据里也是缺失 |
| `has_review_image` | bool | 评论是否带图 | 无缺失 |
| `image_count` | int32 | 评论图片数 | 缺失记 0 |

### 商品侧特征

| 列名 | 类型 | 含义 | 缺失处理 |
|---|---|---|---|
| `has_meta` | bool | 该 `asin` 是否出现在商品表 | 无缺失 |
| `title_clean` | string | 去 HTML 标签、反转义后的标题 | 无商品时缺失 |
| `title_len` | float64 | 清洗后标题字符数 | 无商品时缺失，空标题为 0 |
| `brand_clean` | string | 去掉 `by` 前缀与多余换行后的品牌 | 无商品或原缺失时缺失 |
| `price` | float32 | 解析出的美元价格 | 83.44% 商品缺失；按评论加权缺失 80.71% |
| `price_missing` | bool | 价格是否缺失，含无商品记录 | 无缺失 |
| `log_price` | float32 | `log1p(price)` | 价格缺失时缺失 |
| `main_cat_clean` | string | 反转义、去标签后的主品类 | 无商品或 HTML 碎片时为缺失 |
| `category_leaf` | string | 类目路径最后一段有效名称，跳过 `</span>` 残片 | 无商品或空类目时缺失 |
| `category_depth` | float64 | 有效类目段数 | 无商品时缺失 |
| `also_buy_len` | float64 | `also_buy` 列表长度 | 空列表 0，无商品时缺失 |
| `also_view_len` | float64 | `also_view` 列表长度 | 空列表 0，无商品时缺失 |
| `feature_count` | float64 | 卖点条目数 | 空列表 0，无商品时缺失 |
| `description_len` | float64 | 商品描述字符数 | 无商品或原缺失时缺失 |
| `rank_count` | float64 | 排名条目数 | 空列表 0，无商品时缺失 |
| `image_count_meta` | float64 | 商品图片数 | 空列表 0，无商品时缺失 |
| `tech1_present` | bool | 是否填了 `tech1` | 无商品记 `False` |
| `tech2_present` | bool | 是否填了 `tech2` | 无商品记 `False` |
| `days_since_product` | float32 | 评论日期减商品 `date` 的天数，负值截为 0 | 商品日期缺失时缺失，占 84.63% |

### 时间窗特征

| 列名 | 类型 | 含义 | 缺失处理 |
|---|---|---|---|
| `reviews_prev_30d_same_asin` | int32 | 同一商品按 `unixReviewTime` 排序，当前评论之前 30 天内该商品的评论数，不含当前评论 | 无缺失；首条为 0 |
| `days_since_prev_review_same_asin` | float32 | 距同商品上一条评论的天数 | 每个商品首条评论缺失，占 2.76% |

时间窗只用当前评论之前的同商品记录，不看未来；按 `asin` 分组后计算，训练折与验证折不会共享同一商品的评论。游戏本体范围内 `reviews_prev_30d_same_asin` 均值 30.61、中位 4、p90 61、p99 425、最大 3,585；`days_since_prev_review_same_asin` 中位 4 天、p90 108 天、最大 6,662 天。

### 交互特征

| 列名 | 类型 | 计算方式 | 用途 |
|---|---|---|---|
| `verified_x_log_len` | float32 | `verified × log1p(review_len)` | 已验证用户的长评论是否更偏正向 |
| `log_price_x_verified` | float32 | `log_price × verified` | 价格与购买验证的交互 |
| `helpful_x_len` | float32 | `log1p(vote_count) × log1p(review_len)` | 有用票与文本长度的交互 |
| `days_since_verified` | float32 | `days_since_product × verified` | 商品上市时间与验证购买的交互 |

## 防泄漏约定

- 折先于任何目标相关变换；P4 的 TF-IDF、缺失填补、标准化与特征选择只能在训练折内拟合。
- 不使用 `overall` 的任何聚合作为特征：没有商品平均分、用户平均分、按标签统计的评论数。
- `price_missing`、`has_meta`、`has_style`、`has_vote` 都是缺失指示，不依赖标签。
- 同一 `asin` 只在一个折内，避免商品级目标泄漏。
- `overall` 保留在宽表中仅供审计，不进入 `MODEL_FEATURES`。

## 断言

构建脚本在写出后检查以下条件，任一失败即中断：

- 宽表行数等于评论表行数 2,565,349。
- 标签只含 0/1；游戏本体折号 0–4，范围外折号 -1。
- 游戏本体不少于 100 万条，五折均非空。
- 建模特征 36 个，衍生特征 31 个，均超过表 1 的下限。
- `MODEL_FEATURES` 与 `overall`、`label`、`fold`、`in_scope`、`reviewText`、`summary` 无交集。
- 同一 `asin` 不跨折，五折正类率均在 70%–80%。
- 价格非负，文本长度非负，时间窗计数与间隔非负。

## 已知限制

- 品类边界规则会漏掉部分 Retro Gaming 类目下的老游戏，也会把少量标题未命中硬件词的平台配件当作游戏；规则与示例见 `docs/scope_definition.md`，报告里作为口径限制说明。
- 范围外 1,109,257 条评论保留在宽表中，只用于对照与审计，不进入 P4/P5 的建模样本。
- 15,259 条评论的时间早于商品 `date`，已把 `days_since_product` 截为 0；商品 `date` 的含义更接近上架/收录时间，不宜当精确上市日。
- `price` 按评论加权仍缺失 80.71%，P4 需要缺失指示与训练折内分层填补。
- `category_leaf` 缺失 0.79%，主要来自空类目商品与无商品评论。
- `vote_count=0` 不能区分“真的零票”和“原数据未记录”，解释时只当作结构性缺失处理。
- 有 75,256 组 `(reviewerID, asin, unixReviewTime)` 重复键，可能同一用户同一商品同一秒有多条记录；因按 `asin` 分组，它们仍在同一折，不跨折泄漏。P4 可做一次去重敏感性对照。
