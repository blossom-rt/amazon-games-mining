# 数据字典

统计基于 `src/parse.py` 产出的 `data/processed/reviews.parquet` 与 `data/processed/meta.parquet`，原始文件为 data/raw 下两个 gzip JSON Lines。字段统计由 `results/tables/field_stats.json` 记录，数值均可复现。

统计口径：

- 非空率按解析后的 parquet 列计算，`null` 表示原始记录中该键不存在或值为空。
- 列表字段保留原始顺序与重复项，未做去重。
- 文本长度按字符数计，分位数取 p50、p90、p99。
- 商品表已按 `asin` 去重，保留首次出现的记录。

## 总览

| 表 | 原始记录 | 解析后行数 | 主键 | 用途 |
|---|---|---|---|---|
| 评论表 | 2,565,349 | 2,565,349 | `reviewerID` + `asin` + `unixReviewTime`（候选） | 分类任务、文本特征、时序分析 |
| 商品表 | 84,819 | 71,911 | `asin` | 关联规则、聚类、商品画像 |

商品表原始 84,819 条中有 12,908 组 `asin` 重复，且每组内容逐字节相同；去重后唯一商品 71,911 件。评论表没有重复行，但组合键的唯一性要到一致性检查阶段再验证。

## 评论表

| 字段 | 类型 | 非空率 | 取值范围或长度 | 含义 |
|---|---|---|---|---|
| `reviewerID` | string | 100% | 示例 `A2SUAM1J3GNN3B` | 评论人匿名编号 |
| `asin` | string | 100% | 示例 `0439381673` | 商品编号，与商品表连接键 |
| `reviewerName` | string | 99.99% | 中位 10 字符，最长 1,262 | 评论人昵称 |
| `overall` | float | 100% | 1.0–5.0，均值 4.022，中位 5.0 | 星级评分，分类标签来源 |
| `verified` | bool | 100% | true 占 75.95% | 是否验证购买 |
| `reviewTime` | string | 100% | 示例 `06 9, 2014` | 原始时间字符串，格式不统一 |
| `unixReviewTime` | int64 | 100% | 1997-11-08 至 2018-10-02 | 评论时间戳，时序分析用此列 |
| `reviewText` | string | 99.93% | 中位 151 字符，p90 958，p99 3,879，最长 32,721 | 评论正文，非结构化数据主体 |
| `summary` | string | 99.97% | 中位 16 字符，p90 50，最长 300 | 评论标题 |
| `style` | string | 51.45% | 中位 17 字符，最长 101 | 买家所选规格，原为 dict，已拼成 `键=值; 键=值` |
| `vote` | int32 | 17.26% | 2–5,154，中位 4，均值 8.40 | 认为该评论有用的人数，原为字符串 |
| `image_count` | int32 | 100% | 0–54，非零 28,305 条（1.10%） | 买家晒图数量，原始 `image` 为 URL 列表 |

评分分布：1 星 311,891、2 星 141,333、3 星 212,346、4 星 412,413、5 星 1,487,366。按 `POSITIVE_RATING_THRESHOLD = 4.0` 划分，正向 1,899,779 条（74.06%），负向 665,570 条（25.94%）。

## 商品表

| 字段 | 类型 | 非空率 | 取值范围或长度 | 含义 |
|---|---|---|---|---|
| `asin` | string | 100% | 唯一 | 商品编号 |
| `title` | string | 99.99% | 商品标题 | 商品名称 |
| `brand` | string | 94.97% | 示例 `Coxlures` | 品牌，部分记录带 `by` 前缀与换行 |
| `price_raw` | string | 68.73% | 示例 `$0.72`，也可能是 HTML 片段 | 原始价格字符串，保留用于溯源 |
| `price` | float64 | 16.56% | 0–1,672.15，中位 16.99，均值 38.13，p90 83.86 | 解析后的美元价格，无法解析为空 |
| `main_cat` | string | 99.83% | `Video Games`、`Computers` 等 | 主品类，含 `&amp;` 与少量 HTML 图片串 |
| `category` | list\<string\> | 100%（空列表 2.91%） | 长度中位 4，p90 9，最长 22 | 类目路径，首层几乎都是 `Video Games` |
| `also_buy` | list\<string\> | 100%（空列表 60.13%） | 长度中位 0，p90 57，上限 100 | 常一起购买的商品，关联规则数据源 |
| `also_view` | list\<string\> | 100%（空列表 47.90%） | 长度中位 1，p90 52，上限 60 | 常一起浏览的商品 |
| `feature` | list\<string\> | 100%（空列表 25.23%） | 长度中位 4，p90 5，最长 57 | 卖点列表 |
| `description` | string | 87.49% | 多行文本 | 商品描述，原为 list，已用换行拼接 |
| `rank` | list\<string\> | 100%（空列表 5.89%） | 长度中位 2，最长 4 | 销售排名字符串，含 `&amp;`，需解析数字 |
| `imageURL` | list\<string\> | 100%（空列表 26.29%） | 长度中位 2，p90 8，最长 42 | 主图 URL |
| `imageURLHighRes` | list\<string\> | 100%（空列表 26.29%） | 非空商品 53,007 件 | 高清主图 URL |
| `tech1` | string | 3.93% | 多为空 | 第一张参数表，本品类基本缺失 |
| `tech2` | string | 0.32% | 多为空 | 第二张参数表 |
| `fit` | string | 0% | 全空 | 尺码，本品类不适用 |
| `similar_item` | string | 0.36% | HTML 表格 | 相似商品表 |
| `date` | string | 13.29% | 多为空 | 上架日期 |

主品类分布（去重后）：Video Games 61,076、Computers 6,091、Software 1,519、All Electronics 1,229、Cell Phones & Accessories 424、Toys & Games 362、Home Audio & Theater 139、Amazon Home 139；其余为长尾，另有 120 条 `main_cat` 为空。

`also_buy` 非空商品 28,668 件（39.87%），篮长均值 31.7、中位 15、p90 95、上限 100。

## 与字段字典旧稿的差异

`meta_Video_Games.json.gz` 是 2018 版快照，实际没有以下字段，旧稿里的 `salesRank`、`categories`、`bought_together` 均不存在：

- 销售排名只有 `rank`（字符串列表），没有 `salesRank` 字典
- 类目只有 `category`（列表），没有 `categories` 多维列表
- 共购只有 `also_buy` 与 `also_view`，没有 `bought_together`

`tech1`、`tech2` 实际是字符串而非 dict；`price` 是字符串而非数值；`imageURL` 与 `imageURLHighRes` 是列表。后续代码一律以本表为准。

## 缺失机制初步判断

| 字段 | 缺失率 | 证据 | 判断 |
|---|---|---|---|
| `price` | 83.44% | Video Games 缺失 86.8%，Toys & Games 缺失 49.2%，Computers 缺失 63.9%；缺失率随品类显著变化 | 不是 MCAR，至少是 MAR（依赖品类与商品类型）；可能同时是 MNAR（低价/第三方商品页面不展示价格） |
| `vote` | 82.74% | 有值最小为 2，说明 0 票与 1 票都记为缺失；1999 年非空率 85.2%，2018 年降到 2.2%；1 星评论非空率 32.4%，5 星只有 12.4% | 结构性缺失（没有投票），不是随机丢失；按 MNAR 处理，不能当中位数填补，应保留 `has_vote` 指示变量或补 0 |
| `style` | 48.55% | 1999 年非空率 24.9%，2018 年升到 57.2%；各评分档差异不大（47.6%–52.7%） | 与评论年份强相关，属于 MAR；可按年份分层填补并加缺失指示 |
| `reviewText` | 0.07% | 缺失量极小，各评分档分布接近 | 可近似按 MCAR 处理，直接删除或填空串 |
| `brand` | 5.03% | 缺失集中在长尾品类与非游戏商品 | MAR，依赖品类；可用 `main_cat` 分层众数填补 |
| `description` | 12.51% | 与品类、价格缺失同向 | MAR，保留缺失指示后按品类填补 |
| `category` | 空列表 2.91% | 空列表与 `main_cat` 为空高度重叠 | 结构缺失，按 `main_cat` 回填首层 |

## 表间覆盖与一致性

- 评论表涉及 71,982 个不同 `asin`；商品表 71,911 件中 71,909 件至少有一条评论。
- 3,676 条评论（占 0.14%）的 `asin` 不在商品表中，融合时这些行会失去商品侧特征，需在报告中说明并决定保留或剔除。
- `main_cat` 存在 `&amp;` 未解码、`None` 字符串与 HTML `<img>` 片段；`category` 首层几乎全为 `Video Games`，清洗要下钻到第二、三层。
- `Retro Gaming & Microconsoles` 与 `Retro Gaming &amp; Microconsoles` 两种编码并存，合并时统一解码后再计数。
- `brand` 常见 `"by\n    \n    EA Games"` 形式，需去前缀与空白。
- `price` 的 `price_raw` 去重后约 3.68 万条是 `<div>`、`<span>`、`<script>` 片段，解析脚本只接受以 `$` 开头且后接数字的字符串。

## 治理验证补充

缺失填补对比、异常值检测与一致性复算的完整结果见 [governance_report.md](governance_report.md)，原始数值落在 `results/tables/governance_report.json`。

关键结论：

- 评论表没有单一主键：`(reviewerID, asin, unixReviewTime)` 有 75,256 行重复，分布在 75,048 组，最大一组 70 行；P4 可做一次去重敏感性对照，但不跨折泄漏。
- 价格回填以“品类中位数”MAE 最低（29.97 美元），但 R² 仍为负（-0.055），说明现有商品字段无法可靠还原缺失价格；P4 应保留 `price_missing` 指示，不把回填价格当作强特征。
- 异常值：价格 IQR 检出 1,129 件（9.48%），评论长度 z-score 检出 48,758 条（1.90%），IsolationForest 在 20 万条样本上按 1% 污染率检出 2,000 条；建议长度做 log1p 加 p99 截断，不直接删行。
