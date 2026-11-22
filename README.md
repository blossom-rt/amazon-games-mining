[简体中文](README.zh-CN.md) | English

# Amazon Video Games Data Mining

[![CI](https://github.com/blossom-rt/amazon-games-mining/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/blossom-rt/amazon-games-mining/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An end-to-end data mining project on the Amazon Review Data (2018) Video Games subset. It joins two heterogeneous sources, builds a reproducible pipeline, and covers data governance, association rules, review classification, product clustering, model interpretation, and an interactive dashboard.

- **Reviews**: unstructured text plus ratings, timestamps, and verified-purchase flags.
- **Product metadata**: semi-structured JSON with categories, brand, price, `also_buy`, and `also_view`.

## Documentation

| Topic | Entry |
|---|---|
| Data source, fields, and scope | [data source](docs/en/data_source.md), [data dictionary](docs/en/data_dictionary.md), [feature dictionary](docs/en/feature_dictionary.md), [scope definition](docs/en/scope_definition.md), [scope audit](docs/en/scope_audit.md) |
| Environment and fonts | [setup](docs/en/setup.md) |
| Governance and modeling | [governance report](docs/en/governance_report.md), [classification report](docs/en/classification_report.md), [rules review](docs/en/rules_review.md) |
| Figures | [figure reproduction](docs/en/figure_reproduction.md) |
| Dashboard | [dashboard guide](docs/en/dashboard_guide.md), [deployment](docs/en/streamlit_deploy.md) |
| Full documentation index | [English docs](docs/en/README.md), [Chinese originals](docs/README.md) |

The English documentation is a full translation of the Chinese sources; the Chinese originals are linked from the [Chinese README](README.zh-CN.md).

## Analysis Stages

| Stage | Content |
|---|---|
| P0 | Parse gzip JSON Lines into parquet; fill the field dictionary |
| P1 | Missingness mechanisms, imputation comparison, outliers, consistency checks, and EDA |
| P2 | Review × product wide table, time windows, interaction features, grouped folds |
| P2.5 | Game-product boundary; fixes the P3/P4/P5 sample scope |
| P3 | `also_buy` association rules; Apriori vs FP-Growth |
| P4 | Review sentiment classification; 5-fold and 10-fold evaluation, error analysis, SHAP |
| P5 | Product clustering; elbow, silhouette, bootstrap ARI |
| P6 | `run_all.py`, `04_repro.ipynb`, run logs, result archiving |
| Supplementary | Calibration, embeddings, ablation, word-level terms, time, networks, link prediction, decision curves |
| Dashboard | Streamlit with 4 views and 2 linked filters |

Module ownership: P3 rules, P5 clustering, and the corresponding rule/clustering documents are maintained by [@coldcodefish](https://github.com/coldcodefish); P0–P2.5 data work, P4 classification, P6 reproduction, Streamlit/static dashboards, README, environment, field dictionaries, governance and classification documents are maintained by [@blossom-rt](https://github.com/blossom-rt). `main` is the stable release branch and `dev` is the integration branch.

## Data at a Glance

| Item | Value |
|---|---|
| Reviews | 2,565,349 |
| Raw product records | 84,819 (official count 84,893) |
| Deduplicated products | 71,911 |
| Game-product scope | 40,259 products / 1,456,092 reviews |
| Positive reviews (4–5 stars) | 74.75% |
| Wide table | 2,565,349 × 47; 36 model features, 31 derived features |
| Time range | Nov 1997 – Oct 2018 |

The two source files are placed in `data/raw/` and are not versioned:

```text
Video_Games.json.gz        493.7 MB   reviews
meta_Video_Games.json.gz    50.8 MB   product metadata
```

Checksums, official fields, and citation format are in [data_source](docs/en/data_source.md).

## Key Results

### Association rules

19,032 baskets; `also_buy` keeps only game-product pairs.

| support / confidence | Raw rules | Deduplicated | Apriori | FP-Growth |
|---|---:|---:|---:|---:|
| 0.020 / 0.15 | 0 | 0 | 0.00 s | 0.04 s |
| 0.010 / 0.20 | 42 | 21 | 0.34 s | 0.09 s |
| 0.005 / 0.10 | 1,864 | 932 | 72.76 s | 9.09 s |
| Default (0.005 / 0.20) | 1,855 | 932 | 66.5 s | 8.4 s |

- FP-Growth is about **8× faster** at low support, with an identical rule set.
- 34 of the 932 rules are same-series; median conviction is 3.21.
- High-lift rules concentrate on same-series / hidden-object download editions; cross-series combinations are not stable.

### Review classification

5-fold OOF, grouped by `asin`; no product crosses folds.

| Model | Macro F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|
| Dummy | 0.4277 | 0.7475 | 0.5000 |
| Logistic regression (SGD) | 0.8583 | 0.9838 | 0.9551 |
| Linear SVM (SGD) | **0.8635** | **0.9837** | **0.9569** |
| HistGradientBoosting | 0.7849 | 0.9701 | 0.9160 |
| Random forest | 0.7747 | 0.9673 | 0.9086 |

- 5-fold Wilcoxon `p=0.0625`; the 10-fold replication gives `p=0.001953`, so the 5-fold p-value is limited by sample size.
- Text-only macro F1 is **0.8727**, higher than the combined **0.8635**; metadata-only is 0.5788.
- BGE-small on all 1,456,092 reviews, 10-fold, 300 characters, RTX 4060: macro F1 **0.8658** vs hashing TF-IDF **0.8197**; paired difference 0.0461, 95% CI [0.0419, 0.0506], `p=0.001953`; embedding 1,052 s, total 2,580 s. The 60k sample gives 0.8243 vs 0.7540, same direction.
- Scope note: the main classifier uses 3,000-character hashing TF-IDF and scores 0.8635; full BGE is only 0.0023 higher and costs much more, so it is a supplement for short-text/semantic cases, not a replacement.
- Negation-scope features raise macro F1 from 0.8689 to 0.8785; the negation error rate drops from 15.95% to 14.39%, and contrast errors from 17.64% to 16.35%.

### Product clustering

40,259 game products, 13 clustering features, `k=4`:

- Silhouette **0.209**
- Bootstrap ARI **0.9849** (95% interval 0.9296–0.9976)

| Cluster | Products | Mean rating | Mean reviews | verified | Price observed |
|---|---:|---:|---:|---:|---:|
| 0 Medium popularity, lower rating | 7,797 | 3.641 | 25.4 | 0.700 | 0.0001 |
| 1 High popularity, dense co-purchase | 11,877 | 4.011 | 94.5 | 0.638 | 0.1511 |
| 2 Long tail, sparse co-purchase | 18,430 | 3.747 | 5.9 | 0.662 | 0.1814 |
| 3 Medium popularity, Software | 2,155 | 3.671 | 12.5 | 0.709 | 0.3926 |

The clusters separate mainly by popularity and co-purchase structure, not by rating; names should be read as operational segments rather than natural rating tiers.

### Pipeline runtime

The seven-step full run takes 847.3 s with status `ok`:

| Step | Time |
|---|---:|
| parse | 93.4 s |
| scope | 4.7 s |
| build_features | 109.9 s |
| governance | 15.3 s |
| rules | 101.5 s |
| classify | 492.5 s |
| cluster | 29.6 s |

The run log is written to `results/tables/run_log.json` and records the git commit, timings, hardware, and per-step outputs.

## Supplementary Experiments

Supplementary experiments keep the main seven-step sample scope and probe robustness and business interpretation.

### Robustness

| Experiment | Result | Takeaway |
|---|---|---|
| 10-fold replication | SVM macro F1 0.8642±0.0088; paired difference 0.4365 (95% CI 0.4314–0.4417) | The 5-fold p=0.0625 is not evidence of no difference |
| Probability calibration | ECE 0.088→0.00036; Brier 0.0849→0.0695 | Isotonic calibration clearly improves probability quality |
| Embeddings | Full 300-character text: BGE 0.8658 vs TF-IDF 0.8197; p=0.001953 | Short-text gain; main model is 0.8635, and cost is high |
| Time extrapolation | 2015+ macro F1 0.9028–0.9133; back-test 2003–2006 falls to 0.756–0.776 | Stable after 2015; early data is not the same distribution |
| Modality ablation | Text 0.8727 / metadata 0.5788 / combined 0.8635 | Text is the main signal; metadata adds no stable gain |

### Business interpretation and network structure

| Experiment | Result | Takeaway |
|---|---|---|
| Cluster external validation | 12,078 held-out products; review-count ε²=0.437, also_buy ε²=0.479, rating only 0.006 | Clusters still split by popularity/co-purchase |
| Clustering comparison | GMM 0.1441; HDBSCAN 2 clusters 0.3119; priced subset 0.2102 | None beats KMeans k=4 |
| Three networks | also_buy / also_view / co-reviewers; cross-main-category 1.15% / 1.77% / 2.59% | Recommendation network is narrower than view/review networks |
| Link prediction | Common neighbours Recall@10 0.436, Recall@50 0.725; popularity 0.010 | Recommendation slots are predictable from co-occurrence, but this is not purchase prediction |
| Boundary audit | 226 strong hardware candidates; first 30: 21 hardware, 8 bundles, 1 pure game | Do not collapse the boundary with title-only rules; Retro excludes another 1,154 items |

### Business thresholds

200k sample, 5-fold by `asin`, isotonic calibration:

| Use | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Manual review start (FN=5×FP) | 0.17 | 0.871 | 0.987 | 0.925 |
| High-confidence auto-pass | 0.46 | 0.928 | 0.955 | 0.941 |
| Default | 0.50 | 0.934 | 0.948 | 0.941 |

- At 0.17, net benefit is 0.714; flagging all is 0.694.
- At 0.46, net benefit is 0.666; flagging all is 0.530.
- 0.01 has the highest net benefit but is close to “flag all” and is not suitable as an operational threshold.

## Core Findings

1. **Ratings are driven almost entirely by text.** Text-only macro F1 is 0.8727, above the combined 0.8635; metadata is better for segmentation than for rating prediction.
2. **Negation and contrast are the main error sources.** Error rates are 12.48% for samples with negation and 13.95% for contrast; negation-scope features give only a small gain, while contrast still needs work.
3. **The four clusters are operational segments, not rating tiers.** Cluster 1 is high-popularity and co-purchase dense; cluster 2 is 45.8% long tail; the rating effect size ε² is only 0.006.
4. **`also_buy` is a recommendation-slot network, not a purchase network.** Top rules are same-series / same-platform; the median cross-series support is 0.0059, with no stable cross-category combinations.
5. **Recommendation-slot completion is predictable, but it is not purchase prediction.** Common-neighbour Recall@10 is 0.436, popularity is 0.010, and `also_view` cosine is 0.266.
6. **Classification probabilities fit a “threshold + manual review” workflow.** 0.46 can be a high-confidence auto-pass and 0.17 a review start; low-threshold net benefit is dominated by the 74.6% positive rate.

## Acceptance Extras

| Item | Result | Takeaway |
|---|---|---|
| Nested tuning | Outer 5-fold × inner 3-fold RandomizedSearch; default/tuned macro F1 0.8631/0.8631 | Fixed parameters are already near the plateau; the tuning path works but gives no gain |
| Subgroup errors | Overall 8.87%; 3-star 37.24%, 4-star 13.66%, 5-star 2.67% | 3-star neutral and early reviews are hardest; review them first |
| Imbalance | Original / class_weight / undersampling: LR 0.8778/0.8683/0.8600, SVM 0.8821/0.8702/0.8644 | Reweighting and undersampling both lower macro F1; keep the original distribution plus thresholds |
| Feature selection | 34 → 32 → top15; numeric features F1 0.6205 vs 0.6203 | No gain; first 10 PCA components explain 58.15%, and t-SNE shows class overlap |
| Rule redundancy and memory | Apriori 72.76 s / 15.5 GB; FP-Growth 9.09 s / 27.5 MB; 0 trivial/redundant rules | FP-Growth is better for reproduction; Apriori is near the memory limit |

The field dictionary includes field-level units, sources, and time points; both “or” options in the acceptance table (Notebook vs dashboard, PCA/t-SNE vs feature selection) keep two implementations.

## Interactive Dashboard

`src/dashboard_app.py` uses Streamlit + Plotly and has 4 views:

- **Overview**: sample size, rating, review volume, time-extrapolation F1.
- **Clusters**: size and business metrics of the four operational clusters.
- **Time**: yearly macro F1 and PR-AUC.
- **Network**: cross-category shares and edge overlap of the three networks.

Two sidebar filters:

- **Operational cluster**: links the overview KPI and cluster view.
- **Review year**: links the time KPI and time curve.

Launch:

```bash
pip install -r requirements-dashboard.txt
streamlit run src/dashboard_app.py --server.port 8501
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501); the health check is `/_stcore/health`.

Public entry points:

- **Static interactive version (no server, recommended stable entry)**: [open static dashboard](https://raw.githack.com/blossom-rt/amazon-games-mining/main/results/dashboard.html). The page has overview, clusters, time, and network tabs; query parameters such as `?tab=overview`, `?tab=clusters`, `?tab=time`, `?tab=network` open a tab directly. The sidebar cluster multiselect and review-year slider are the two linked filters.
- **Streamlit public demo (free public app)**: [open Streamlit app](https://amazon-games-mining-dashboard.streamlit.app/). It runs on Streamlit Community Cloud from `main`, entry `src/dashboard_app.py`.

The free public Streamlit app sleeps after about 12 hours without visits. When “This app has gone to sleep due to inactivity” appears, click “Yes, get this app back up!”; data and code are not lost. See [dashboard_guide](docs/dashboard_guide.md) for screenshots and deployment notes; prefer the static version when a wake-up-free entry is needed.

## Repository Layout

```text
amazon-games-mining/
├── data/
│   ├── raw/                     two .json.gz source files, not committed
│   └── processed/               parquet, scope, folds, not committed
├── src/
│   ├── config.py                paths, seed, thresholds, figure fonts
│   ├── download.py              download and SHA256 verification
│   ├── parse.py                 gzip JSON → parquet
│   ├── scope.py                 game-product boundary and in_scope
│   ├── build_features.py        wide table, time windows, folds
│   ├── governance.py            missingness, outliers, consistency
│   ├── rules.py                 association rules and threshold sensitivity
│   ├── classify.py              main classifier and evaluation
│   ├── classify_*.py            robustness, ablation, terms, time, negation, decision, tuning, subgroups, imbalance
│   ├── figure_pipeline.py       unified redraw and validation for 44 figures
│   ├── cluster*.py              main clustering, comparison, insights, validation
│   ├── network_compare.py       also_buy / also_view / co-reviewers
│   ├── link_prediction.py       recommendation-slot link prediction
│   ├── dashboard_app.py         Streamlit dashboard
│   └── run_all.py               one-command reproduction
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_rules.ipynb
│   ├── 03_cluster.ipynb
│   └── 04_repro.ipynb
├── results/
│   ├── figures/                 fig01–fig50, 44 files
│   └── tables/                  summaries, metrics, errors, run_log
├── docs/
│   ├── en/                      full English translations
│   └── *.md                     Chinese originals
├── .github/workflows/ci.yml
├── requirements-core.txt
├── requirements.txt
├── requirements-dashboard.txt
├── requirements-optional.txt
├── README.md
└── README.zh-CN.md
```

## Environment

```bash
conda create -n amazon-games python=3.11 -y
conda activate amazon-games
pip install -r requirements.txt
python -m ipykernel install --user --name amazon-games --display-name "Python (amazon-games)"
```

If the default pip mirror fails, add `-i https://pypi.org/simple`.

Chinese labels in figures need a CJK font; Windows usually has `SimHei`, and other systems are covered in [setup](docs/en/setup.md).

## Branches and Collaboration

| Branch | Purpose | Rule |
|---|---|---|
| `main` | Stable release | Merge via PR only |
| `dev` | Integration | Daily development baseline |
| `feat/*` | Short-lived feature branch | Cut from `dev`, PR back to `dev` |

Work lands on `dev`; feature branches are named by stage and never push to `main` directly. For a release, open a PR from `dev` to `main`, then tag the merge on `main`.

Start new work:

```bash
git switch dev
git pull --ff-only
git switch -c feat/<stage>
```

Commit:

```bash
git add <files>
git commit -m "feat(<stage>): ..."
git push -u origin feat/<stage>
```

Feature branches merge into `dev` via PR; `data/`, `__pycache__`, and `.ipynb_checkpoints` are not versioned.

Before release, run `python src/run_all.py --check --extras` and CI on `dev`, then PR into `main` and tag `vX.Y.Z`.

## Commands

### Data preparation

```bash
# 1. Download two gzip files and verify SHA256
python src/download.py
python src/download.py --verify

# 2. Parse, define scope, build wide table and folds
python src/parse.py
python src/scope.py
python src/build_features.py
```

Existing outputs are not overwritten by default; add `--force` to regenerate, e.g. `python src/parse.py --force`. Outputs are written to `data/processed/`:

```text
reviews.parquet  meta.parquet  scope.parquet
features.parquet  folds.parquet
```

### Main pipeline

Run governance → rules → classification → clustering:

```bash
python src/governance.py
python src/rules.py
python src/classify.py
python src/cluster.py
```

Each script writes summaries/CSVs to `results/tables/` and figures to `results/figures/`. To run a subset, use `run_all.py` with `--only` / `--skip`:

```bash
python src/run_all.py --only parse,scope,build_features
python src/run_all.py --skip classify
```

### Supplementary experiments

```bash
python src/classify_robustness.py
python src/classify_ablation.py
python src/classify_terms.py
python src/classify_time.py --limit 300000
python src/classify_negation.py --limit 150000
python src/classify_decision.py
python src/classify_tuning.py
python src/classify_subgroups.py
python src/classify_imbalance.py
python src/feature_selection.py
python src/network_compare.py
python src/link_prediction.py --sources 2000
python src/cluster_compare.py
python src/cluster_insights.py
python src/cluster_validate.py
python src/scope_audit.py
```

The embedding comparison is optional and needs extra dependencies plus a first-time model download:

```bash
pip install -r requirements-optional.txt
export HF_ENDPOINT=https://hf-mirror.com
python src/classify_embedding.py --full --text-chars 300 --folds 10 --backend sentence-transformers --device cuda
```

On PowerShell use `$env:HF_ENDPOINT = "https://hf-mirror.com"`. On CPU use `--backend fastembed --device cpu`; the full `sentence-transformers` comparison is too slow without a GPU.

### Dashboard and reproduction notebook

```bash
# Streamlit dashboard
pip install -r requirements-dashboard.txt
streamlit run src/dashboard_app.py --server.port 8501

# Rebuild the static HTML dashboard
python src/build_static_dashboard.py

# Reproduction notebook
jupyter lab notebooks/04_repro.ipynb
```

Streamlit opens [http://127.0.0.1:8501](http://127.0.0.1:8501); the static version is `results/dashboard.html`. `04_repro.ipynb` reads existing artifacts by default; set `RUN_FULL_PIPELINE=1` to trigger a full rerun.

### One-command reproduction

| Command | Behavior |
|---|---|
| `python src/run_all.py --check` | Check required artifacts only; no retraining |
| `python src/run_all.py --check --extras` | Also check supplementary artifacts |
| `python src/run_all.py --check --require-all` | Treat missing optional steps as failure; final acceptance |
| `python src/run_all.py --extras` | Full main pipeline, supplementary experiments, redraw/validate 44 figures |
| `python src/run_all.py --only parse,scope,build_features` | Run selected steps |
| `python src/run_all.py --skip classify` | Skip selected steps |

The main pipeline takes about 14 minutes; `--extras` also runs link prediction, decision curves, tuning, subgroup errors, imbalance, feature selection, and time extrapolation, then the `figures` step. `results/tables/run_log.json` records commands, timings, hardware, outputs, and git state.

### Figure reproduction

All 44 result figures are redrawn and validated by `src/figure_pipeline.py`. Full path:

```bash
python src/run_all.py --extras
```

With existing `data/processed/` and `results/tables/`, redraw the 33 table-driven figures and validate all 44:

```bash
python src/figure_pipeline.py --all
python src/figure_pipeline.py --check
```

The other 11 figures (fig20/21/23/25, 30–33, 43, 49, 50) are written directly by `cluster.py`, `classify.py`, `classify_time.py`, `classify_subgroups.py`, and `feature_selection.py`; `--all` reports the corresponding command when one is missing. See [figure reproduction](docs/en/figure_reproduction.md) for inputs, fonts, and missing-table handling.

### CI

`.github/workflows/ci.yml` runs on push/PR to `dev` or `main`, and on manual dispatch:

1. Install `requirements-core.txt` (without Streamlit/Plotly to keep CI fast).
2. Run `python src/figure_pipeline.py --check` to validate the 44 committed figures.
3. Cache and verify `data/raw`.
4. Run `parse → scope → build_features`.
5. Run `python src/run_all.py --check --require-all --only parse,scope,build_features`.

CI does not train rules, classification, clustering, or supplementary models; run `python src/run_all.py --check --extras` locally for full acceptance. CI checks figure integrity only and does not redraw; full redraw remains the last step of `--extras`. `data/processed/` is not committed, so CI rebuilds it from cached raw data.

## Reproducibility Conventions

- Random steps use `SEED = 42` (`src/config.py`).
- Run from the project root, or use `python -m src.<script>`.
- `data/processed/` and `*.parquet` are not committed; generate them with the data-preparation steps. `results/tables/` and `results/figures/` are committed as lightweight inputs for figure reproduction.

## Known Limitations

- Price is missing for 83.44%; all four imputations have negative R². Price is used only for post-hoc profiling, not as a strong feature.
- `also_buy` is a recommendation slot, not a real purchase basket; rules and link prediction describe recommendation structure only.
- Of 226 strong hardware-word candidates, 8 of the first 30 are bundles; the main scope does not use title-only hard filters.
- The BGE comparison uses full data but truncates to 300 characters; the main model uses 3,000-character TF-IDF and full BGE is only 0.0023 higher, so this is not evidence that embeddings replace TF-IDF.
- Cluster silhouette is 0.209, indicating weak separation; the clusters are better treated as operational segments than natural classes.

## Contributing

Issues and pull requests are welcome:

1. Branch from `dev` using `feat/<topic>` or `fix/<topic>`.
2. Do not commit `data/`, `*.parquet`, report files, or machine-specific paths.
3. Before opening a PR, run `python src/run_all.py --check --extras`, `python src/figure_pipeline.py --check`, and `python -m py_compile src/*.py`.
4. Target `dev` and describe the purpose, scope, and validation commands.

## License

Code is released under the [MIT License](LICENSE). The Amazon Review Data is subject to its own citation and usage terms; see [data source](docs/en/data_source.md).
