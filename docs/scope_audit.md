# 游戏本体边界复核

## 目的

`scope.py` 用类目路径把“游戏本体”与配件、主机、DLC 分开，但 `software_node` 分支仍可能混入标题写明硬件的商品。本复核只回答两个问题：

1. 标题含强硬件词、但被 `software_node` 保留的商品有多少，是否应一刀切排除；
2. Retro 类目下因缺少 Games 节点被排除的商品有多少，是否误伤游戏本体。

本复核不自动改样本口径。是否收紧由人工判断，避免把“游戏 + 手柄同捆”误删。

## 强硬件词候选

筛选条件：

- `in_scope=True` 且 `reason=software_node`；
- 标题命中 `controller|gamepad|headset|console|memory card|charger|power cord|adapter|stylus|dock|repair|replacement|dance pad|joystick|remote`；
- 不含 `case|cover|stand` 这类常见游戏标题词，避免 “Jewel Case”“Deep Cover” 大量误报。

结果：226 件，7,105 条评论，占范围内评论 0.49%。完整清单在 `results/tables/scope_audit_candidates.csv`。

## 评论量前 30 的人工抽查

逐条查看标题与类目路径，按“商品主体”分为四类：

| # | ASIN | 标题（截断） | 判断 |
|---:|---|---|---|
| 1 | B000KRXAGE | Wii Play with Wii Remote | 游戏+手柄同捆 |
| 2 | B004WLRQCI | Wii Console with Mario Kart Wii Bundle | 主机同捆 |
| 3 | B000K6TOK6 | Sony Playstation 1 COMPLETE System Console | 主机 |
| 4 | B00129I75I | Mizar Replacement AV Cables + AC Power Cord | 纯硬件 |
| 5 | B006PP4136 | PlayStation Vita Starter Kit with Memory Card | 纯硬件 |
| 6 | B000I4JIK6 | Guitar Hero II: Game & Guitar Controller Bundle | 游戏+外设同捆 |
| 7 | B00NAW5UZA | Nintendo New 3DS XL Battery Charging Dock | 纯硬件 |
| 8 | B000ZKA0KA | SOCOM bundled with Bluetooth Headset | 游戏+耳机同捆 |
| 9 | B003O6EETU | Xbox 360 250GB Halo Reach Console Bundle | 主机同捆 |
| 10 | B002I0I85Y | FlingSmash with Wii Remote Plus | 游戏+手柄同捆 |
| 11 | B000WJFZEA | Warhawk (No Headset) | 纯游戏 |
| 12 | B000GHG0BA | DDR 1" Foam Deluxe Dance Pad | 纯硬件 |
| 13 | B007WRGNQE | Nintendo 3DS Portable Gaming Console | 主机 |
| 14 | B000F7V2GO | Replacement Joystick | 纯硬件 |
| 15 | B00006FWUU | PS2 DualShock 2 Controller | 纯硬件 |
| 16 | B017KIHHH8 | Rock Band 4 Wireless Guitar Controller | 纯硬件 |
| 17 | B014CLMJ74 | Rock Band 4 Bundle with Legacy Adapter | 游戏+外设同捆 |
| 18 | B000K9K9AC | WarHawk Bundle with Bluetooth Headset | 游戏+耳机同捆 |
| 19 | B000VRAUAC | Wii Hooked! Real Motion Fishing w/controller | 游戏+手柄同捆 |
| 20 | B00FFHEZTA | Mario & Sonic with Blue Wii Remote Plus | 游戏+手柄同捆 |
| 21 | B00L0J7DUK | Bluray Drive Replacement for Nintendo | 纯硬件 |
| 22 | B00012LZXS | PlayStation 2 DVD Remote Control | 纯硬件 |
| 23 | B00006FWUT | PS2 DualShock 2 Controller - Emerald | 纯硬件 |
| 24 | B0050SZDPE | N64 USB Controller for PC/Mac | 纯硬件 |
| 25 | B0000520GG | PlayStation 2 Vertical Console Stand | 纯硬件 |
| 26 | B01FANMZE0 | FastSnail Nunchuck + Remote Controller Set | 纯硬件 |
| 27 | B00025LDT0 | Redoctane Ignition Dance Pad | 纯硬件 |
| 28 | B00S674UQE | Wii/Wii U Remote Plus Controller - Toad | 纯硬件 |
| 29 | B01DHJ56TU | FastSnail Nunchuck + Remote Controller Set | 纯硬件 |
| 30 | B00255V7JQ | Slim AC Adapter for Playstation 2 | 纯硬件 |

统计：纯硬件/主机 21 件，游戏+配件同捆 8 件，纯游戏 1 件。

## 结论

1. 不能按“标题含硬件词”一刀切。前 30 条中有 8 条是游戏 + 手柄/耳机同捆，直接排除会删掉真实游戏评论。
2. 可以只排除“纯硬件/主机”子集，但这需要逐条判断，或增加“标题以硬件名词结尾且不含游戏名”的规则；后者仍会误伤。
3. 更稳妥的做法是保留默认 `scope.py`，在分类特征中增加 `is_bundle` / `hardware_title` 标记，由模型自己学习；或在敏感性分析中报告“剔除 226 件”的结果。
4. 若做敏感性分析，剔除全部 226 件后：商品 40,033、评论 1,448,987，分别少 0.56% 与 0.49%。这个量级不足以改变主结论，但应在口径说明中交代。

## Retro 缺 Games 节点

`Retro Gaming` 类目下共 14,486 件，其中 `no_software_node` 被排除 1,154 件、17,109 条评论。此前文档中的“904 件”是更早的子集口径；按当前 `scope.py` 是 1,154 件。

被排除的并不都是配件，例如：

- `Grand Theft Auto San Andreas`（PS2 游戏，但路径缺 Games 节点）；
- `Dance Dance Revolution Dance Pad`（纯硬件）；
- `Sony PSone Memory Card`（配件）。

因此不能把 Retro 的 `no_software_node` 全部放回。建议：

- 对 1,154 件做标题+平台人工抽查，把确认是游戏本体的 ASIN 加入白名单；
- 或在口径文档中明确“Retro 缺节点商品保守排除”，并把 1,154 件/17,109 条评论作为已知限制，而不是逐条修复。

## 产物

- `results/tables/scope_audit_candidates.csv`：226 件候选，含 `review_verdict`、`review_note` 空列，供人工填写；
- `results/tables/scope_audit_retro.csv`：1,154 件 Retro 排除清单；
- `results/tables/scope_audit_summary.json`：计数与影响估计。

## 建议的下一步

1. 先不改变默认口径，避免全量结果与既有数字全部重跑；
2. 在口径说明中引用本复核，把“275 件”改为“226 件强硬件词候选，人工抽查前 30 条中 21 件纯硬件/主机、8 件同捆”；
3. 如果要做敏感性实验，在 `scope.py` 增加 `--strict-hardware` 开关，单独跑一版 `features_strict.parquet` 和分类指标，不覆盖主结果。
