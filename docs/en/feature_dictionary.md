# Feature Dictionary and Wide-Table Specification

## Outputs

| File | Size | Description |
|---|---|---|
| `data/processed/features.parquet` | 2,565,349 rows × 47 columns, 497.3 MB | Review × product wide table, one row per review, including `in_scope` |
| `data/processed/folds.parquet` | 1,456,092 rows × 4 columns, 13.8 MB | Game products only: `reviewerID`, `asin`, `label`, `fold` |
| `results/tables/feature_summary.json` | — | Fold distribution, missing rates, runtime, and hardware |

Build command:

```bash
python src/build_features.py --force
```

A full build took 106.4 seconds on Windows 10 with 28 cores, 15.7 GB of memory, Python 3.11.18, and pyarrow 26.0.1.

## Join Specification

- The review table (2,565,349 rows) is left-joined to the product table (71,911 rows) on `asin`, validated as `many_to_one`.
- The product table is deduplicated by `asin`; 12,908 exact duplicate records among the original 84,819 records do not enter the wide table.
- The category boundary is decided by `src/scope.py`: of 71,911 products, 40,259 are game products, covering 1,456,092 reviews (56.76%); the rules and exclusion reasons are in [scope definition](./scope_definition.md).
- The wide table keeps all reviews, and `in_scope` marks whether a review enters the modeling sample; `folds.parquet` contains only rows with `in_scope=True`.
- For 3,676 reviews the `asin` is absent from the product table; these have `has_meta=False`, `in_scope=False`, and product fields that are missing values or `False`; these reviews are not dropped.
- The wide table keeps the raw `reviewText` and `summary`; text cleaning and vectorization are deferred to P4 and fitted within the training folds.

## Labels and Folds

- Label definition: `label = 1` if and only if `overall >= 4.0`; across the full data there are 1,899,779 positives (74.06%), and the positive rate for game products is 74.75%.
- Fold assignment: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, with grouping key `asin`, fitted only on the 1,456,092 reviews with `in_scope=True`.
- Folds are generated before any encoding, imputation, standardization, or feature selection; later scripts only read the `fold` column and may not repartition.
- Reviews of the same `asin` fall in a single fold, so reviews of one product never appear in both training and validation.
- Out-of-scope reviews remain in the wide table with `fold = -1` and do not enter modeling or fold statistics.

| Fold | Rows | Positive rate |
|---|---|---|
| 0 | 291,217 | 74.75% |
| 1 | 291,221 | 74.75% |
| 2 | 291,204 | 74.75% |
| 3 | 291,208 | 74.75% |
| 4 | 291,242 | 74.75% |

## Field Table

### Identifiers and Target

| Column | Type | Meaning | Missing-value handling |
|---|---|---|---|
| `reviewerID` | string | Reviewer ID | No missing values |
| `asin` | string | Product ID | No missing values |
| `unixReviewTime` | int64 | Review timestamp, seconds | No missing values |
| `overall` | float32 | Raw rating 1–5, audit only | No missing values |
| `label` | int8 | Target variable, 1 when `overall >= 4` | No missing values |
| `fold` | int8 | Stratified group fold number 0–4; -1 outside scope | No missing values |
| `in_scope` | bool | Whether the product belongs to the game-product scope, decided by `src/scope.py` | No missing values |

### Review Text

| Column | Type | Meaning | Missing-value handling |
|---|---|---|---|
| `reviewText` | string | Raw review body | 1,720 missing, nulls retained |
| `summary` | string | Raw review title | 822 missing, nulls retained |

### Review-Side Features

| Column | Type | Meaning | Missing-value handling |
|---|---|---|---|
| `verified` | bool | Whether the purchase is verified | No missing values |
| `review_year` | int16 | Review year, 1997–2018 | No missing values |
| `review_month` | int8 | Review month 1–12 | No missing values |
| `review_dow` | int8 | Day of week, 0 = Monday | No missing values |
| `review_len` | int32 | Body character count | Empty text recorded as 0 |
| `review_word_count` | int32 | Word count split on whitespace | Empty text recorded as 0 |
| `review_sentence_count` | int32 | Number of segments delimited by `[.!?]` plus 1 | Empty text recorded as 0 |
| `summary_len` | int32 | Title character count | Empty title recorded as 0 |
| `has_style` | bool | Whether `style` variant information is present | No missing values |
| `has_vote` | bool | Whether the helpful-vote field is present | No missing values |
| `vote_count` | int32 | Helpful vote count | Missing recorded as 0; note that 0/1 votes are also missing in the raw data |
| `has_review_image` | bool | Whether the review has images | No missing values |
| `image_count` | int32 | Number of review images | Missing recorded as 0 |

### Product-Side Features

| Column | Type | Meaning | Missing-value handling |
|---|---|---|---|
| `has_meta` | bool | Whether the `asin` appears in the product table | No missing values |
| `title_clean` | string | Title after removing HTML tags and unescaping | Missing when there is no product |
| `title_len` | float64 | Character count of the cleaned title | Missing when there is no product; empty title is 0 |
| `brand_clean` | string | Brand after removing the `by` prefix and extra line breaks | Missing when there is no product or the original is missing |
| `price` | float32 | Parsed USD price | Missing for 83.44% of products; 80.71% when weighted by review |
| `price_missing` | bool | Whether the price is missing, including records with no product | No missing values |
| `log_price` | float32 | `log1p(price)` | Missing when the price is missing |
| `main_cat_clean` | string | Main category after unescaping and tag removal | Missing when there is no product or only an HTML fragment |
| `category_leaf` | string | Last valid name in the category path, skipping `</span>` fragments | Missing when there is no product or the category is empty |
| `category_depth` | float64 | Number of valid category segments | Missing when there is no product |
| `also_buy_len` | float64 | Length of the `also_buy` list | Empty list is 0; missing when there is no product |
| `also_view_len` | float64 | Length of the `also_view` list | Empty list is 0; missing when there is no product |
| `feature_count` | float64 | Number of bullet-point features | Empty list is 0; missing when there is no product |
| `description_len` | float64 | Character count of the product description | Missing when there is no product or the original is missing |
| `rank_count` | float64 | Number of ranking entries | Empty list is 0; missing when there is no product |
| `image_count_meta` | float64 | Number of product images | Empty list is 0; missing when there is no product |
| `tech1_present` | bool | Whether `tech1` is populated | Recorded as `False` when there is no product |
| `tech2_present` | bool | Whether `tech2` is populated | Recorded as `False` when there is no product |
| `days_since_product` | float32 | Review date minus the product `date`, in days; negative values clipped to 0 | Missing when the product date is missing, which is 84.63% |

### Time-Window Features

| Column | Type | Meaning | Missing-value handling |
|---|---|---|---|
| `reviews_prev_30d_same_asin` | int32 | For the same product ordered by `unixReviewTime`, the number of reviews of that product in the 30 days before the current review, excluding the current review | No missing values; the first review is 0 |
| `days_since_prev_review_same_asin` | float32 | Days since the previous review of the same product | Missing for the first review of each product, which is 2.76% |

Time windows use only same-product records before the current review and never look ahead; they are computed after grouping by `asin`, so training and validation folds never share reviews of the same product. Within the game-product scope, `reviews_prev_30d_same_asin` has mean 30.61, median 4, p90 61, p99 425, and maximum 3,585; `days_since_prev_review_same_asin` has median 4 days, p90 108 days, and maximum 6,662 days.

### Interaction Features

| Column | Type | Computation | Purpose |
|---|---|---|---|
| `verified_x_log_len` | float32 | `verified × log1p(review_len)` | Whether long reviews from verified users skew more positive |
| `log_price_x_verified` | float32 | `log_price × verified` | Interaction between price and verified purchase |
| `helpful_x_len` | float32 | `log1p(vote_count) × log1p(review_len)` | Interaction between helpful votes and text length |
| `days_since_verified` | float32 | `days_since_product × verified` | Interaction between time since product listing and verified purchase |

## Leakage-Prevention Conventions

- Folds precede any target-related transformation; TF-IDF, missing-value imputation, standardization, and feature selection in P4 may only be fitted within the training folds.
- No aggregate of `overall` is used as a feature: there is no product mean rating, user mean rating, or label-conditioned review count.
- `price_missing`, `has_meta`, `has_style`, and `has_vote` are all missingness indicators and do not depend on the label.
- Each `asin` lies in exactly one fold, preventing product-level target leakage.
- `overall` is kept in the wide table for audit only and does not enter `MODEL_FEATURES`.

## Assertions

After writing the outputs, the build script checks the following conditions and aborts if any fails:

- The wide table row count equals the review table row count, 2,565,349.
- Labels contain only 0/1; game products have fold numbers 0–4 and out-of-scope reviews have fold number -1.
- Game products number at least 1,000,000, and none of the five folds is empty.
- There are 36 modeling features and 31 derived features, both above the lower bound in Table 1.
- `MODEL_FEATURES` has no overlap with `overall`, `label`, `fold`, `in_scope`, `reviewText`, or `summary`.
- No `asin` crosses folds, and the positive rate of all five folds lies within 70%–80%.
- Prices are non-negative, text lengths are non-negative, and time-window counts and gaps are non-negative.

## Known Limitations

- The category boundary rules miss some older games under Retro Gaming categories, and also treat a few platform accessories whose titles lack hardware words as games; the rules and examples are in [scope definition](./scope_definition.md), and the limitations of the scope are in [scope audit](./scope_audit.md).
- The 1,109,257 out-of-scope reviews remain in the wide table, used only for comparison and audit; they do not enter the P4/P5 modeling sample.
- 15,259 reviews predate the product `date`, and `days_since_product` has been clipped to 0 for them; the product `date` is closer to a listing/catalog time and should not be treated as an exact release date.
- `price` is still 80.71% missing when weighted by review; P4 needs missingness indicators and stratified imputation within the training folds.
- `category_leaf` is 0.79% missing, mainly from products with empty categories and reviews without a product.
- `vote_count=0` cannot distinguish "genuinely zero votes" from "not recorded in the raw data"; treat it as structural missingness when interpreting results.
- There are 75,256 duplicate `(reviewerID, asin, unixReviewTime)` keys, so the same user may have several records for the same product in the same second; because grouping is by `asin`, they still fall in the same fold and do not leak across folds. P4 may run a deduplication sensitivity comparison.
