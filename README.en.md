[简体中文](README.md) | English

# Amazon Video Games Data Mining

[![CI](https://github.com/blossom-rt/amazon-games-mining/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/blossom-rt/amazon-games-mining/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An end-to-end data mining project on the Amazon Review Data (2018) Video Games subset. It joins two heterogeneous sources, builds a reproducible pipeline, and covers data governance, association rules, review classification, product clustering, model interpretation, and an interactive dashboard.

- **Reviews**: unstructured text plus ratings, timestamps, and verified-purchase flags.
- **Product metadata**: semi-structured JSON with categories, brand, price, `also_buy`, and `also_view`.

Detailed documentation is written in Chinese. This English README is a quick-start guide; see [README.md](README.md) for the full documentation index.

## Data at a Glance

| Item | Value |
|---|---|
| Reviews | 2,565,349 |
| Raw product records | 84,819 |
| Deduplicated products | 71,911 |
| Game-product scope | 40,259 products / 1,456,092 reviews |
| Positive reviews (4–5 stars) | 74.75% |
| Wide table | 2,565,349 × 47; 36 model features, 31 derived features |
| Time range | Nov 1997 – Oct 2018 |

Raw files are not committed. Download and verify them with:

```bash
python src/download.py
python src/download.py --verify
```

Checksums, field definitions, and the dataset citation are in [docs/data_source.md](docs/data_source.md).

## Results

| Task | Result |
|---|---|
| Review classification | Linear SVM, 5-fold macro F1 **0.8635**, PR-AUC **0.9837**, ROC-AUC **0.9569** |
| 10-fold replication | Macro F1 **0.8642**, paired test p = 0.00195 |
| Text vs metadata | Text only **0.8727**, metadata only **0.5788**, combined **0.8635** |
| BGE-small comparison | 300-character text: BGE **0.8658** vs hashing TF-IDF **0.8197**; main 3,000-character model is **0.8635**, so BGE is not adopted |
| Association rules | 19,032 baskets; 1,855 raw / 932 deduplicated rules; FP-Growth is much faster than Apriori |
| Product clustering | KMeans k=4; silhouette **0.209**; bootstrap ARI **0.9849** |
| Time extrapolation | Forward 2014→2015+ macro F1 **0.9048**; backward **0.8249** |
| Temporal / subgroup errors | 3-star reviews and early-year reviews are the hardest groups |

The main classification model uses hashing TF-IDF plus a linear SVM. Calibration, subgroup errors, link prediction, decision curves, and network comparisons are included as supplementary experiments.

## Quick Start

```bash
conda create -n amazon-games python=3.11 -y
conda activate amazon-games
pip install -r requirements.txt
```

Reproduce the full pipeline, including supplementary experiments and figures:

```bash
python src/run_all.py --extras
python src/run_all.py --check --extras
```

Run only the main pipeline:

```bash
python src/run_all.py
python src/run_all.py --check
```

Run the dashboard:

```bash
streamlit run src/dashboard_app.py --server.port 8501
```

A static fallback is available at `results/dashboard.html`.

## Figures

All 44 result figures have a single entry point:

```bash
python src/figure_pipeline.py --check
python src/figure_pipeline.py --all
```

- `--check` verifies that all 44 committed images exist, are readable, and meet the minimum size.
- `--all` regenerates table-driven figures and validates the full set. Figure 34/35 use the fitted classification artifacts, so run the classification steps first.
- CI runs only `--check`; it does not retrain models or redraw figures.

See [docs/figure_reproduction.md](docs/figure_reproduction.md) for the figure-to-input mapping.

## Repository Layout

```text
src/                 pipeline, experiments, dashboard, figure entry point
results/figures/     44 committed figures
results/tables/      JSON/CSV result tables used by figures and dashboard
docs/                Chinese documentation and data dictionaries
notebooks/           EDA, rules, clustering, and reproduction notebooks
```

## Reproducibility

- Random steps use `SEED = 42`.
- `data/processed/` and `*.parquet` are not committed; they are rebuilt from raw data.
- `results/tables/` and `results/figures/` are committed as lightweight inputs for figures.
- Paths in result JSON files are repository-relative, so the repository works from any clone location.

## Contributing

Issues and pull requests are welcome:

1. Branch from `dev` using `feat/<topic>` or `fix/<topic>`.
2. Do not commit raw data, parquet files, reports, or machine-specific paths.
3. Before opening a PR, run:
   ```bash
   python src/run_all.py --check --extras
   python src/figure_pipeline.py --check
   python -m py_compile src/*.py
   ```
4. Target `dev` and describe the purpose, scope, and validation commands.

## License

Code is released under the [MIT License](LICENSE). The Amazon Review Data is subject to its own citation and usage terms; see [docs/data_source.md](docs/data_source.md).
