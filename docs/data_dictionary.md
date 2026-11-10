# 数据字典

本文件在数据解析完成后填写实际统计值。当前表格中的「待统计」占位符由 `notebooks/01_eda.ipynb` 输出后回填。

统计口径说明：

- 非空率按「键是否存在」计算，不按「是否为空字符串」计算。原始数据中部分字段在记录里直接不出现，与字段存在但值为空是两回事。
- 类型按解析到 Python 对象后的实际类型统计，同一字段出现多种类型时全部列出。
- 取值范围只对数值与日期字段给出，文本字段给出长度分位数。

## 评论表 Video_Games.json.gz

记录数 2,565,349，主键为（reviewerID, asin, unixReviewTime）的组合。

| 字段 | 类型 | 非空率 | 取值范围或示例 | 含义 |
|---|---|---|---|---|
| `reviewerID` | str | 待统计 | `A2SUAM1J3GNN3B` | 评论人匿名编号 |
| `asin` | str | 待统计 | `0000013714` | 商品编号，与元数据表连接键 |
| `reviewerName` | str | 待统计 | `A. McDonald` | 评论人昵称 |
| `overall` | float | 待统计 | 1.0 — 5.0 | 星级评分，分类任务标签来源 |
| `reviewText` | str | 待统计 | 长文本 | 评论正文，非结构化数据主体 |
| `summary` | str | 待统计 | `Heavenly Highway Hymns` | 评论标题 |
| `verified` | bool | 待统计 | `true` | 是否已验证购买 |
| `vote` | str 或 int | 待统计 | `2` | 认为该评论有用的人数，类型不统一 |
| `style` | dict | 待统计 | `{"Size:": "Large"}` | 买家所选规格，键缺失率高 |
| `image` | list | 待统计 | URL 列表 | 买家晒图 |
| `unixReviewTime` | int | 待统计 | `1252800000` | 评论时间戳，时序分析用此字段 |
| `reviewTime` | str | 待统计 | `09 13, 2009` | 原始时间字符串，格式需解析 |

## 元数据表 meta_Video_Games.json.gz

记录数 84,819，主键为 `asin`。

| 字段 | 类型 | 非空率 | 取值范围或示例 | 含义 |
|---|---|---|---|---|
| `asin` | str | 待统计 | `0000031852` | 商品编号 |
| `title` | str | 待统计 | 商品标题 | 商品名称 |
| `brand` | str | 待统计 | `Coxlures` | 品牌，部分记录带 `by` 前缀与换行 |
| `price` | str | 待统计 | `$0.72` | 价格，大量为空或 HTML 残留 |
| `main_cat` | str | 待统计 | `Video Games` | 主品类，部分含 HTML 实体 |
| `category` | list | 待统计 | `["Video Games", "PC", "Games"]` | 类目路径，尾部可能残留标签 |
| `categories` | list | 待统计 | 多维列表 | 类目，与 `category` 并存 |
| `feature` | list | 待统计 | 五点描述 | 卖点列表 |
| `description` | list 或 str | 待统计 | 详情描述 | 商品描述 |
| `rank` | list | 待统计 | `">#67231 in Video Games"` | 销售排名，字符串需解析 |
| `salesRank` | dict | 待统计 | `{"Video Games": 211836}` | 销售排名，与 `rank` 并存 |
| `also_buy` | list | 待统计 | ASIN 列表 | 常一起购买的商品，关联规则数据源 |
| `also_view` | list | 待统计 | ASIN 列表 | 常一起浏览的商品 |
| `bought_together` | list | 待统计 | ASIN 列表 | 组合购买 |
| `tech1` | dict 或 str | 待统计 | 技术参数表 | 第一张参数表 |
| `tech2` | dict 或 str | 待统计 | 技术参数表 | 第二张参数表 |
| `similar_item` | str | 待统计 | HTML 表格 | 相似商品表 |
| `imageURL` | list | 待统计 | URL | 主图 |
| `imageURLHighRes` | list | 待统计 | URL | 高清主图 |
| `fit` | str | 待统计 | 空 | 尺码，本品类基本为空 |
| `date` | str | 待统计 | 空 | 上架日期，基本为空 |

## 待补充的统计项

以下内容在解析完成后追加：

1. 各字段非空率与类型分布
2. `also_buy` 非空商品数、篮长分布的均值、中位数、分位数、上限
3. 评分分布，以及按 `POSITIVE_RATING_THRESHOLD` 划分后的类别比例
4. 品类路径第一层与第二层的取值分布，用于品类边界清洗
5. 重复记录数与去重规则
6. 缺失机制判定结论（MCAR / MAR / MNAR）与依据
