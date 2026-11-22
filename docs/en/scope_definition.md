# Category Boundary Definition (P2.5)

## Purpose

Amazon's Video Games category mixes in accessories, controllers, consoles, repair parts, and DLC. The P3 association rules, P4 classification, and P5 clustering all need a reviewable game-product scope; otherwise the rules would treat accessories as game pairings, and classification would mix hardware reviews into game reviews.

This definition uses only product category paths and titles. It reads no ratings, review counts, or any other target variable, so no label information enters the sample-selection stage.

## Inputs and Outputs

| Item | Path |
|---|---|
| Input product table | `data/processed/meta.parquet` (71,911 rows) |
| Input wide table | `data/processed/features.parquet` (`asin` and `overall` are read only for coverage statistics) |
| Output scope table | `data/processed/scope.parquet` (asin, in_scope, reason, path_depth, path_text) |
| Output summary | `results/tables/scope_summary.json` |

Run:

```bash
python src/scope.py --force
```

## Category Path Cleaning

The raw `category` is a list of product categories and commonly contains two kinds of dirty data:

- The last segment is a `</span></span></span>` fragment, for example `['Video Games', 'PC', '', '</span></span></span>']`;
- The last segment is a product-description sentence, potentially hundreds of characters long.

`effective_path` scans in the original order: empty strings are skipped, and the scan is truncated at a segment containing `</span`, containing `none`/`null`, or longer than 90 characters, keeping the valid segments already read. This way a fragment is never taken as the leaf and a description sentence is never taken as a category.

## Decision Order

Each product is evaluated in order and stops at the first match:

1. Effective path is empty → `empty_category`;
2. Root node is not `Video Games` → `non_video_games`;
3. Path contains `Accessor` (matching `Accessories`, `Accessory Kits`, and so on) → `accessories`;
4. The leaf matches the hardware list or a hardware word (controller, headset, console, case, cable, stand, figure, and so on) → `hardware_leaf`;
5. Path contains `Downloadable Content` → `dlc`;
6. Path contains `Games` or `Digital Games` → `software_node`, kept;
7. Only two levels and the second level is a platform (`Video Games > PC`, `Video Games > PlayStation 4`, and so on), with no hardware word in the title → `platform_only`, kept;
8. Only two levels but the title matches a hardware word → `hardware_title`, excluded;
9. Any other path → `no_software_node`, excluded.

`software_node` and `platform_only` count as `in_scope=True`; all other reasons are out of scope.

Title hardware-word regex:

```text
case|cover|skin|protector|screen|cable|cord|charger|battery|
controller|gamepad|headset|mouse|keyboard|stand|mount|dock|stylus|
adapter|repair|replacement|memory card|dance pad|console|
headphone|microphone|joystick|remote
```

## Results

| Metric | Value |
|---|---|
| Total products | 71,911 |
| Game products | 40,259 (55.98%) |
| Out-of-scope products | 31,652 |
| Total reviews | 2,565,349 |
| Game-product reviews | 1,456,092 (56.76%) |
| Out-of-scope reviews | 1,109,257 |
| Positive rate, game products | 74.75% |
| Positive rate, out of scope | 73.15% |

Distribution of exclusion reasons:

| Reason | Products |
|---|---|
| `software_node` | 38,907 |
| `accessories` | 25,894 |
| `empty_category` | 2,091 |
| `hardware_leaf` | 1,564 |
| `platform_only` | 1,352 |
| `no_software_node` | 1,312 |
| `dlc` | 437 |
| `hardware_title` | 354 |

Main categories of game products: Video Games 38,104; Software 1,466; Toys & Games 167; All Electronics 161; Computers 68; the rest are long-tail and missing.

Most common in-scope paths:

| Path | Products |
|---|---|
| Video Games > PC > Games | 15,762 |
| Video Games > Retro Gaming & Microconsoles > PlayStation 2 > Games | 1,879 |
| Video Games > Kids & Family > Nintendo DS > Games | 1,595 |
| Video Games > Xbox 360 > Games | 1,262 |
| Video Games > PlayStation 3 > Games | 1,161 |

## Boundary Cases

Kept:

- `Video Games > PC` + `Urban Runner`: platform path, no hardware word in the title.
- `Video Games > PlayStation 4` + `Mafia 3 (PS4)`: platform path.
- `Video Games > Retro Gaming & Microconsoles > PlayStation 2 > Games`: explicit `Games` node.

Excluded:

- `Video Games > Wii U` + `YTTL New Touch Screen Repair Replacement Part`: title matches repair/screen, recorded as `hardware_title`.
- `Video Games > PC` + `Dance Dance Revolution Non-Slip Dance Pad`: title matches dance pad, recorded as `hardware_title`.
- Controllers, headsets, and carrying cases under `Video Games > Accessories`: `accessories`.
- `Video Games > Xbox 360 > Digital Games & DLC > Downloadable Content`: `dlc`.

## Known Limitations

- The 1,154 products under `Retro Gaming & Microconsoles` that lack a `Games` node are excluded, corresponding to 17,109 reviews; some of them may be older games, but a conservative rule is used to avoid counting mini consoles and retro accessories.
- Two-level platform paths rely on title filtering, so a few accessories whose titles omit hardware words may slip into scope; classification error analysis will discuss specific samples.
- Among `software_node` products, 226 have titles containing strong hardware words such as controller, console, or cord, corresponding to 7,105 reviews (0.49%); a manual review of the 30 most-reviewed found 21 pure hardware/consoles, 8 game + accessory bundles, and 1 pure game. By default, no secondary filtering by title is applied, to avoid wrongly deleting bundle products.
- The rules use no price, brand, or rating; they answer "does the category and title look like a game", not "is the product actually a game".
- The scope table is decided per product, so all reviews of a product share the same `in_scope`, and a product never has only some reviews in scope.
