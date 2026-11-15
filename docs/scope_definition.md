# 品类边界定义（P2.5）

## 目的

Amazon 的 Video Games 类目混入配件、手柄、主机、维修件与 DLC。P3 关联规则、P4 分类与 P5 聚类都需要一个可复核的“游戏本体”口径，否则规则会把配件当成游戏搭配，分类也会把硬件评论混进游戏评论。

本定义只使用商品类目路径与标题，不读取评分、评论数或任何目标变量，避免在样本筛选阶段引入标签信息。

## 输入与输出

| 项目 | 路径 |
|---|---|
| 输入商品表 | `data/processed/meta.parquet`（71,911 行） |
| 输入宽表 | `data/processed/features.parquet`（只读 `asin`、`overall` 做覆盖统计） |
| 输出范围表 | `data/processed/scope.parquet`（asin、in_scope、reason、path_depth、path_text） |
| 输出摘要 | `results/tables/scope_summary.json` |

运行：

```bash
python src/scope.py --force
```

## 类目路径清洗

原始 `category` 是商品类目列表，常见两种脏数据：

- 末段是 `</span></span></span>` 残片，例如 `['Video Games', 'PC', '', '</span></span></span>']`；
- 末段是一段商品描述句，长度可达数百字符。

`effective_path` 按原顺序扫描：遇到空串继续，遇到含 `</span`、`none`/`null` 或长度超过 90 字符的段就截断，之前已读到的有效段保留。这样既不会把残片当成叶子，也不会把描述句当成类目。

## 判定顺序

对每件商品依次判断，命中即停止：

1. 有效路径为空 → `empty_category`；
2. 根节点不是 `Video Games` → `non_video_games`；
3. 路径含 `Accessor`（匹配 `Accessories`、`Accessory Kits` 等）→ `accessories`；
4. 叶子命中硬件清单或硬件词（controller、headset、console、case、cable、stand、figure 等）→ `hardware_leaf`；
5. 路径含 `Downloadable Content` → `dlc`；
6. 路径含 `Games` 或 `Digital Games` → `software_node`，保留；
7. 仅两层且第二层是平台（`Video Games > PC`、`Video Games > PlayStation 4` 等），标题不含硬件词 → `platform_only`，保留；
8. 仅两层但标题命中硬件词 → `hardware_title`，排除；
9. 其余路径 → `no_software_node`，排除。

`software_node` 与 `platform_only` 计入 `in_scope=True`；其余原因均为范围外。

标题硬件词正则：

```text
case|cover|skin|protector|screen|cable|cord|charger|battery|
controller|gamepad|headset|mouse|keyboard|stand|mount|dock|stylus|
adapter|repair|replacement|memory card|dance pad|console|
headphone|microphone|joystick|remote
```

## 结果

| 指标 | 数值 |
|---|---|
| 商品总数 | 71,911 |
| 游戏本体商品 | 40,259（55.98%） |
| 范围外商品 | 31,652 |
| 评论总数 | 2,565,349 |
| 游戏本体评论 | 1,456,092（56.76%） |
| 范围外评论 | 1,109,257 |
| 游戏本体正类率 | 74.75% |
| 范围外正类率 | 73.15% |

排除原因分布：

| 原因 | 商品数 |
|---|---|
| `software_node` | 38,907 |
| `accessories` | 25,894 |
| `empty_category` | 2,091 |
| `hardware_leaf` | 1,564 |
| `platform_only` | 1,352 |
| `no_software_node` | 1,312 |
| `dlc` | 437 |
| `hardware_title` | 354 |

游戏本体商品的主品类：Video Games 38,104、Software 1,466、Toys & Games 167、All Electronics 161、Computers 68，其余为长尾与缺失。

最常见的在范围路径：

| 路径 | 商品数 |
|---|---|
| Video Games > PC > Games | 15,762 |
| Video Games > Retro Gaming & Microconsoles > PlayStation 2 > Games | 1,879 |
| Video Games > Kids & Family > Nintendo DS > Games | 1,595 |
| Video Games > Xbox 360 > Games | 1,262 |
| Video Games > PlayStation 3 > Games | 1,161 |

## 边界案例

保留：

- `Video Games > PC` + `Urban Runner`：平台路径，标题不含硬件词。
- `Video Games > PlayStation 4` + `Mafia 3 (PS4)`：平台路径。
- `Video Games > Retro Gaming & Microconsoles > PlayStation 2 > Games`：显式 `Games` 节点。

排除：

- `Video Games > Wii U` + `YTTL New Touch Screen Repair Replacement Part`：标题命中 repair/screen，记为 `hardware_title`。
- `Video Games > PC` + `Dance Dance Revolution Non-Slip Dance Pad`：标题命中 dance pad，记为 `hardware_title`。
- `Video Games > Accessories` 下的手柄、耳机、收纳包：`accessories`。
- `Video Games > Xbox 360 > Digital Games & DLC > Downloadable Content`：`dlc`。

## 已知限制

- `Retro Gaming & Microconsoles` 下缺少 `Games` 节点的 1,154 件商品被排除，对应 17,109 条评论；其中可能包含老游戏，但为避免把迷你主机与复古配件算进来，采用保守口径。
- 仅两层的平台路径依赖标题过滤，少量标题未写硬件词的配件可能漏进范围；分类错误分析中会结合具体样本讨论。
- `software_node` 类目中有 226 件商品标题含 controller、console、cord 等强硬件词，对应 7,105 条评论（0.49%）；人工抽查评论量前 30 条，其中 21 件为纯硬件/主机、8 件为游戏+配件同捆、1 件为纯游戏。默认不按标题二次过滤，以免误删同捆商品。
- 规则没有使用价格、品牌或评分；它回答的是“商品类目与标题像不像游戏”，不是“商品实际上是不是游戏”。
- 范围表按商品判定，同一商品的所有评论共享 `in_scope`，不会出现同商品部分评论在范围内的情况。
