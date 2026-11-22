# Data Dictionary

The statistics are based on `data/processed/reviews.parquet` and `data/processed/meta.parquet` produced by `src/parse.py`; the raw files are two gzip JSON Lines files under data/raw. Field statistics are recorded in `results/tables/field_stats.json`, and all values are reproducible.

Statistical conventions:

- Non-null rates are computed on the parsed parquet columns; `null` means the key is absent from the raw record or its value is empty.
- List fields keep their original order and duplicates; no deduplication is applied.
- Text length is counted in characters; quantiles are p50, p90, and p99.
- The product table has been deduplicated by `asin`, keeping the first occurrence.

## Overview

| Table | Raw records | Parsed rows | Primary key | Purpose |
|---|---|---|---|---|
| Review table | 2,565,349 | 2,565,349 | `reviewerID` + `asin` + `unixReviewTime` (candidate) | Classification task, text features, time-series analysis |
| Product table | 84,819 | 71,911 | `asin` | Association rules, clustering, product profiling |

Of the 84,819 raw records in the product table, 12,908 `asin` groups are duplicated, and the content of each group is byte-for-byte identical; after deduplication there are 71,911 unique products. The review table has no duplicate rows, but the uniqueness of the composite key is verified later, in the consistency-check stage.

## Review Table

| Field | Type | Non-null rate | Value range or length | Meaning |
|---|---|---|---|---|
| `reviewerID` | string | 100% | example `A2SUAM1J3GNN3B` | Anonymized reviewer ID |
| `asin` | string | 100% | example `0439381673` | Product ID, join key to the product table |
| `reviewerName` | string | 99.99% | median 10 characters, max 1,262 | Reviewer nickname |
| `overall` | float | 100% | 1.0–5.0, mean 4.022, median 5.0 | Star rating, source of the classification label |
| `verified` | bool | 100% | true in 75.95% | Whether the purchase was verified |
| `reviewTime` | string | 100% | example `06 9, 2014` | Raw time string, format not uniform |
| `unixReviewTime` | int64 | 100% | 1997-11-08 to 2018-10-02 | Review timestamp; time-series analysis uses this column |
| `reviewText` | string | 99.93% | median 151 characters, p90 958, p99 3,879, max 32,721 | Review body, the bulk of the unstructured data |
| `summary` | string | 99.97% | median 16 characters, p90 50, max 300 | Review title |
| `style` | string | 51.45% | median 17 characters, max 101 | Variant selected by the buyer; originally a dict, now flattened to `key=value; key=value` |
| `vote` | int32 | 17.26% | 2–5,154, median 4, mean 8.40 | Number of people who found the review helpful; originally a string |
| `image_count` | int32 | 100% | 0–54, 28,305 non-zero rows (1.10%) | Number of buyer photos; the raw `image` is a list of URLs |

Rating distribution: 1 star 311,891, 2 stars 141,333, 3 stars 212,346, 4 stars 412,413, 5 stars 1,487,366. With `POSITIVE_RATING_THRESHOLD = 4.0`, 1,899,779 reviews are positive (74.06%) and 665,570 are negative (25.94%).

## Product Table

| Field | Type | Non-null rate | Value range or length | Meaning |
|---|---|---|---|---|
| `asin` | string | 100% | unique | Product ID |
| `title` | string | 99.99% | product title | Product name |
| `brand` | string | 94.97% | example `Coxlures` | Brand; some records carry a `by` prefix and line breaks |
| `price_raw` | string | 68.73% | example `$0.72`, possibly an HTML fragment | Raw price string, kept for traceability |
| `price` | float64 | 16.56% | 0–1,672.15, median 16.99, mean 38.13, p90 83.86 | Parsed price in USD; empty when unparseable |
| `main_cat` | string | 99.83% | `Video Games`, `Computers`, etc. | Main category; contains `&amp;` and a few HTML image strings |
| `category` | list\<string\> | 100% (2.91% empty lists) | length median 4, p90 9, max 22 | Category path; the first level is almost always `Video Games` |
| `also_buy` | list\<string\> | 100% (60.13% empty lists) | length median 0, p90 57, cap 100 | Products frequently bought together, the data source for association rules |
| `also_view` | list\<string\> | 100% (47.90% empty lists) | length median 1, p90 52, cap 60 | Products frequently viewed together |
| `feature` | list\<string\> | 100% (25.23% empty lists) | length median 4, p90 5, max 57 | List of selling points |
| `description` | string | 87.49% | multi-line text | Product description; originally a list, joined with newlines |
| `rank` | list\<string\> | 100% (5.89% empty lists) | length median 2, max 4 | Sales-rank strings containing `&amp;`; the numbers must be parsed out |
| `imageURL` | list\<string\> | 100% (26.29% empty lists) | length median 2, p90 8, max 42 | Main image URLs |
| `imageURLHighRes` | list\<string\> | 100% (26.29% empty lists) | 53,007 products non-empty | High-resolution main image URLs |
| `tech1` | string | 3.93% | mostly empty | First specification table, essentially missing for this category |
| `tech2` | string | 0.32% | mostly empty | Second specification table |
| `fit` | string | 0% | entirely empty | Size; not applicable to this category |
| `similar_item` | string | 0.36% | HTML table | Similar-items table |
| `date` | string | 13.29% | mostly empty | Listing date |

Main-category distribution (after deduplication): Video Games 61,076, Computers 6,091, Software 1,519, All Electronics 1,229, Cell Phones & Accessories 424, Toys & Games 362, Home Audio & Theater 139, Amazon Home 139; the rest form a long tail, and a further 120 rows have an empty `main_cat`.

Products with a non-empty `also_buy`: 28,668 (39.87%); basket length mean 31.7, median 15, p90 95, cap 100.

## Differences from the Old Field Dictionary

`meta_Video_Games.json.gz` is the 2018 snapshot and does not actually contain the following fields; `salesRank`, `categories`, and `bought_together` in the old draft do not exist:

- Sales rank exists only as `rank` (a list of strings); there is no `salesRank` dictionary
- Categories exist only as `category` (a list); there is no multi-level `categories` list
- Co-purchase exists only as `also_buy` and `also_view`; there is no `bought_together`

`tech1` and `tech2` are actually strings rather than dicts; `price` is a string rather than numeric; `imageURL` and `imageURLHighRes` are lists. All downstream code takes this table as authoritative.

## Preliminary Assessment of Missingness Mechanisms

| Field | Missing rate | Evidence | Assessment |
|---|---|---|---|
| `price` | 83.44% | Video Games missing 86.8%, Toys & Games 49.2%, Computers 63.9%; the missing rate varies markedly by category | Not MCAR, at least MAR (dependent on category and product type); possibly also MNAR (low-price/third-party product pages do not display a price) |
| `vote` | 82.74% | The smallest non-missing value is 2, implying that both 0 and 1 votes are recorded as missing; non-null rate 85.2% in 1999, dropping to 2.2% in 2018; 32.4% for 1-star reviews but only 12.4% for 5-star | Structural missingness (no votes cast), not random loss; treat as MNAR, do not impute with the median — keep a `has_vote` indicator or fill with 0 |
| `style` | 48.55% | Non-null rate 24.9% in 1999, rising to 57.2% in 2018; little difference across rating bands (47.6%–52.7%) | Strongly related to review year, hence MAR; can be imputed stratified by year with a missing indicator |
| `reviewText` | 0.07% | Extremely small missing volume; distributions across rating bands are close | Can be treated as approximately MCAR; drop directly or fill with an empty string |
| `brand` | 5.03% | Missingness concentrated in long-tail categories and non-game products | MAR, dependent on category; can be imputed with the mode stratified by `main_cat` |
| `description` | 12.51% | Moves in the same direction as missingness in category and price | MAR; keep a missing indicator and impute by category |
| `category` | 2.91% empty lists | Empty lists overlap heavily with an empty `main_cat` | Structural missingness; backfill the first level from `main_cat` |

## Cross-Table Coverage and Consistency

- The review table covers 71,982 distinct `asin` values; of the 71,911 products in the product table, 71,909 have at least one review.
- For 3,676 reviews (0.14%), the `asin` is not in the product table; these rows lose their product-side features during the merge, so the analysis notes must state whether they are retained or dropped.
- `main_cat` contains undecoded `&amp;`, the string `None`, and HTML `<img>` fragments; the first `category` level is almost entirely `Video Games`, so cleaning must drill down to the second and third levels.
- `Retro Gaming & Microconsoles` and `Retro Gaming &amp; Microconsoles` coexist as two encodings; decode uniformly before counting when merging.
- `brand` commonly appears in the form `"by\n    \n    EA Games"`, so the prefix and whitespace must be stripped.
- For `price`, after deduplication about 36,800 distinct `price_raw` values are `<div>`, `<span>`, or `<script>` fragments; the parsing script accepts only strings that start with `$` followed by a digit.

## Modeling Sample Boundary

`src/scope.py` decides whether a product is a game title from the category path and title, without reading ratings: of 71,911 products, 40,259 are in scope (55.98%), covering 1,456,092 reviews (56.76%), with an in-scope positive rate of 74.75% and 73.15% outside the scope. The rules, exclusion reasons, and boundary cases are in [scope_definition.md](./scope_definition.md); the scope table lands in `data/processed/scope.parquet`, the wide table is marked with `in_scope`, and `folds.parquet` contains only in-scope reviews.

## Governance Validation Supplement

Full results for the imputation comparison, outlier detection, and consistency recomputation are in [governance_report.md](./governance_report.md), and the raw values are in `results/tables/governance_report.json`.

Key conclusions:

- The review table has no single primary key: `(reviewerID, asin, unixReviewTime)` has 75,256 duplicate rows distributed across 75,048 groups, the largest group with 70 rows; P4 can run one deduplication sensitivity control, but must not leak across folds.
- Price backfilling has the lowest MAE with the "category median" (USD 29.97), but R² is still negative (-0.055), showing that the available product fields cannot reliably restore missing prices; P4 should keep the `price_missing` indicator and not treat backfilled prices as a strong feature.
- Outliers: the price IQR flags 1,129 products (9.48%), the review-length z-score flags 48,758 reviews (1.90%), and IsolationForest flags 2,000 rows in a 200,000-row sample at a 1% contamination rate; the recommendation is log1p on length plus p99 clipping rather than direct row deletion.

## Field-Level Units, Provenance, and Time Reference

Common source: the Video Games subset of the UCSD Amazon Review Data (2018), downloaded on 2026-11-09; review fields come from `Video_Games.json.gz` and product fields from `meta_Video_Games.json.gz`. Review-type fields are timestamped at review occurrence (`unixReviewTime`, UTC), while product-type fields are timestamped at the 2018 snapshot; price has no historical version and represents only the displayed price at the time the snapshot was captured.

Review table:

| Field | Unit | Source | Time reference |
|---|---|---|---|
| `reviewerID` | none (anonymous ID) | review JSON | review record |
| `asin` | none (product ID) | review JSON | review record |
| `reviewerName` | characters | review JSON | review record |
| `overall` | stars (1–5) | review JSON | review record |
| `verified` | boolean | review JSON | review record |
| `reviewTime` | raw date text | review JSON | review record, format not uniform |
| `unixReviewTime` | Unix seconds (UTC) | review JSON | review occurrence time |
| `reviewText` | characters | review JSON | review record |
| `summary` | characters | review JSON | review record |
| `style` | text (key=value) | `style` dict in the review JSON | review record |
| `vote` | votes (people) | `vote` string in the review JSON | review record; 0/1 votes recorded as missing |
| `image_count` | images | `image` list in the review JSON | review record |

Product table:

| Field | Unit | Source | Time reference |
|---|---|---|---|
| `asin` | none (product ID) | product JSON | 2018 snapshot |
| `title` | characters | product JSON | 2018 snapshot |
| `brand` | characters | product JSON | 2018 snapshot |
| `price_raw` | raw USD string | product JSON | 2018 snapshot |
| `price` | USD | parsed from `price` in the product JSON | 2018 snapshot, not a historical price |
| `main_cat` | category name | product JSON | 2018 snapshot |
| `category` | category level (segment) | product JSON | 2018 snapshot |
| `also_buy` | number of products | product JSON | 2018 snapshot recommendation slots |
| `also_view` | number of products | product JSON | 2018 snapshot recommendation slots |
| `feature` | items | product JSON | 2018 snapshot |
| `description` | characters | product JSON | 2018 snapshot |
| `rank` | rank (string, no unit) | product JSON | 2018 snapshot |
| `imageURL` | number of URLs | product JSON | 2018 snapshot |
| `imageURLHighRes` | number of URLs | product JSON | 2018 snapshot |
| `tech1` | text | product JSON | 2018 snapshot |
| `tech2` | text | product JSON | 2018 snapshot |
| `fit` | text | product JSON | 2018 snapshot, entirely empty |
| `similar_item` | text/HTML | product JSON | 2018 snapshot |
| `date` | date text | product JSON | product listing date, not the snapshot time |
