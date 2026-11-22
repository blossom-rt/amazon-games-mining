# Game-Product Boundary Audit

## Purpose

`scope.py` uses category paths to separate game products from accessories, consoles, and DLC, but the `software_node` branch can still include products whose titles explicitly state hardware. This audit answers only two questions:

1. How many products with strong hardware words in the title are kept by `software_node`, and should they be excluded across the board;
2. How many products under the Retro categories are excluded for lacking a Games node, and does this wrongly remove game products.

This audit does not change the sample scope automatically. Whether to tighten it is a manual decision, to avoid wrongly deleting "game + controller bundle" items.

## Strong-Hardware-Word Candidates

Selection criteria:

- `in_scope=True` and `reason=software_node`;
- Title matches `controller|gamepad|headset|console|memory card|charger|power cord|adapter|stylus|dock|repair|replacement|dance pad|joystick|remote`;
- Excludes common game-title words such as `case|cover|stand`, to avoid many false positives like "Jewel Case" and "Deep Cover".

Result: 226 products, 7,105 reviews, 0.49% of in-scope reviews. The complete list is in `results/tables/scope_audit_candidates.csv`.

## Manual Review of the Top 30 by Review Count

Titles and category paths were reviewed one by one and grouped into four classes by the product's main subject:

| # | ASIN | Title (truncated) | Verdict |
|---:|---|---|---|
| 1 | B000KRXAGE | Wii Play with Wii Remote | Game + controller bundle |
| 2 | B004WLRQCI | Wii Console with Mario Kart Wii Bundle | Console bundle |
| 3 | B000K6TOK6 | Sony Playstation 1 COMPLETE System Console | Console |
| 4 | B00129I75I | Mizar Replacement AV Cables + AC Power Cord | Pure hardware |
| 5 | B006PP4136 | PlayStation Vita Starter Kit with Memory Card | Pure hardware |
| 6 | B000I4JIK6 | Guitar Hero II: Game & Guitar Controller Bundle | Game + peripheral bundle |
| 7 | B00NAW5UZA | Nintendo New 3DS XL Battery Charging Dock | Pure hardware |
| 8 | B000ZKA0KA | SOCOM bundled with Bluetooth Headset | Game + headset bundle |
| 9 | B003O6EETU | Xbox 360 250GB Halo Reach Console Bundle | Console bundle |
| 10 | B002I0I85Y | FlingSmash with Wii Remote Plus | Game + controller bundle |
| 11 | B000WJFZEA | Warhawk (No Headset) | Pure game |
| 12 | B000GHG0BA | DDR 1" Foam Deluxe Dance Pad | Pure hardware |
| 13 | B007WRGNQE | Nintendo 3DS Portable Gaming Console | Console |
| 14 | B000F7V2GO | Replacement Joystick | Pure hardware |
| 15 | B00006FWUU | PS2 DualShock 2 Controller | Pure hardware |
| 16 | B017KIHHH8 | Rock Band 4 Wireless Guitar Controller | Pure hardware |
| 17 | B014CLMJ74 | Rock Band 4 Bundle with Legacy Adapter | Game + peripheral bundle |
| 18 | B000K9K9AC | WarHawk Bundle with Bluetooth Headset | Game + headset bundle |
| 19 | B000VRAUAC | Wii Hooked! Real Motion Fishing w/controller | Game + controller bundle |
| 20 | B00FFHEZTA | Mario & Sonic with Blue Wii Remote Plus | Game + controller bundle |
| 21 | B00L0J7DUK | Bluray Drive Replacement for Nintendo | Pure hardware |
| 22 | B00012LZXS | PlayStation 2 DVD Remote Control | Pure hardware |
| 23 | B00006FWUT | PS2 DualShock 2 Controller - Emerald | Pure hardware |
| 24 | B0050SZDPE | N64 USB Controller for PC/Mac | Pure hardware |
| 25 | B0000520GG | PlayStation 2 Vertical Console Stand | Pure hardware |
| 26 | B01FANMZE0 | FastSnail Nunchuck + Remote Controller Set | Pure hardware |
| 27 | B00025LDT0 | Redoctane Ignition Dance Pad | Pure hardware |
| 28 | B00S674UQE | Wii/Wii U Remote Plus Controller - Toad | Pure hardware |
| 29 | B01DHJ56TU | FastSnail Nunchuck + Remote Controller Set | Pure hardware |
| 30 | B00255V7JQ | Slim AC Adapter for Playstation 2 | Pure hardware |

Tally: 21 pure hardware/consoles, 8 game + accessory bundles, 1 pure game.

## Conclusions

1. A blanket rule based on "hardware words in the title" is not acceptable. 8 of the top 30 are game + controller/headset bundles, and excluding them outright would delete genuine game reviews.
2. Excluding only the "pure hardware/console" subset is possible, but it requires case-by-case judgement, or a rule such as "the title ends with a hardware noun and contains no game name"; the latter would still produce false positives.
3. A more robust approach is to keep the default `scope.py`, add `is_bundle` / `hardware_title` flags to the classification features, and let the model learn; alternatively, report the "remove 226 products" result in a sensitivity analysis.
4. For a sensitivity analysis, removing all 226 products leaves 40,033 products and 1,448,987 reviews, 0.56% and 0.49% fewer respectively. This magnitude is not enough to change the main conclusions, but it should be disclosed in the scope documentation.

## Retro Products Missing a Games Node

The `Retro Gaming` categories contain 14,486 products, of which 1,154 products and 17,109 reviews are excluded by `no_software_node`.

The excluded items are not all accessories, for example:

- `Grand Theft Auto San Andreas` (a PS2 game, but the path lacks a Games node);
- `Dance Dance Revolution Dance Pad` (pure hardware);
- `Sony PSone Memory Card` (accessory).

So the Retro `no_software_node` items cannot all be restored. Recommendations:

- Manually review the 1,154 products by title and platform, and add confirmed game-product ASINs to a whitelist;
- Or state in the scope documentation that "Retro products missing a node are conservatively excluded", and record the 1,154 products / 17,109 reviews as a known limitation rather than fixing them one by one.

## Outputs

- `results/tables/scope_audit_candidates.csv`: 226 candidates, with empty `review_verdict` and `review_note` columns for manual entry;
- `results/tables/scope_audit_retro.csv`: list of the 1,154 excluded Retro products;
- `results/tables/scope_audit_summary.json`: counts and impact estimates.

## Processing Principles

1. Keep the default scope unchanged, to avoid rerunning all analyses for 0.49% boundary samples;
2. Cite these audit results in the scope documentation: 226 strong-hardware-word candidates, and in the manually reviewed top 30, 21 pure hardware/consoles and 8 bundles;
3. If a sensitivity analysis is needed later, add a strict mode to `scope.py` that writes `features_strict.parquet` separately without overwriting the main results.
