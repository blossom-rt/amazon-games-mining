# Data Source

## Dataset

The project uses the **Video Games** subset of [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html), released by UCSD.

| File | Size | Content |
|---|---:|---|
| `Video_Games.json.gz` | 493.7 MB | Review records, one JSON object per line |
| `meta_Video_Games.json.gz` | 50.8 MB | Product metadata, one JSON object per item |

Raw files belong in `data/raw/` and are not committed. Download and verify them with:

```bash
python src/download.py
python src/download.py --verify
```

The downloader writes to a temporary `.part` file, verifies SHA256, and only then replaces the target. The repository records the expected checksums in `src/config.py`; the full checksum values and field-level details are in the [Chinese data source document](../data_source.md).

## Scope

- Reviews: 2,565,349 records after parsing.
- Product metadata: 84,819 raw records, 71,911 unique ASINs after exact-record deduplication.
- Game-product scope: 40,259 products and 1,456,092 reviews, using category paths and title rules.
- The scope fixes the P3/P4/P5 sample. Products outside the scope are kept for audits but excluded from rules, classification, and clustering.

## Fields Used

Reviews: `overall`, `reviewText`, `summary`, `unixReviewTime`, `verified`, `vote`, `style`, `reviewerID`, `asin`.

Products: `asin`, `title`, `brand`, `category`, `price`, `also_buy`, `also_view`, `description`, `feature`, `main_cat`, `tech1`.

See the [data dictionary](../data_dictionary.md) and [feature dictionary](../feature_dictionary.md) for missing rates, cleaning rules, and derived fields.

## Citation and Terms

If you use the data, cite the original dataset paper:

> Ni J., Li J., McAuley J. Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects. EMNLP 2019.

The dataset is provided for research use; follow the terms on the official dataset page. This repository does not redistribute the raw files. Reviewer names and reviewer IDs are not exposed in the dashboard or aggregated outputs.
