# Setup

## Requirements

- Python 3.11 (3.11.x tested)
- About 2 GB free disk for raw data, parquet intermediates, and figures
- A CJK font if you regenerate figures with Chinese labels

```bash
conda create -n amazon-games python=3.11 -y
conda activate amazon-games
pip install -r requirements.txt
```

`requirements.txt` includes the analysis stack, Streamlit, and Plotly. For analysis only:

```bash
pip install -r requirements-core.txt
```

For the dashboard only:

```bash
pip install -r requirements-dashboard.txt
```

If the default package mirror fails, add `-i https://pypi.org/simple`.

## Fonts

Figures use Chinese labels. The plotting helpers search for:

```text
Microsoft YaHei → SimHei → Noto Sans CJK SC → PingFang SC
```

Windows usually has `SimHei`; on Linux install `fonts-noto-cjk` or provide a CJK font. Missing glyphs appear as empty boxes. If that happens, install the font and rebuild the figures rather than changing the data.

## Data

```bash
python src/download.py
python src/download.py --verify
python src/parse.py
python src/scope.py
python src/build_features.py
```

Outputs are written to `data/processed/`: `reviews.parquet`, `meta.parquet`, `scope.parquet`, `features.parquet`, and `folds.parquet`. Existing outputs are protected; add `--force` to overwrite.

## Dashboard

```bash
streamlit run src/dashboard_app.py --server.port 8501
```

Open <http://127.0.0.1:8501>; health check: <http://127.0.0.1:8501/_stcore/health>.

The static fallback is `results/dashboard.html`.

## Optional Embedding Comparison

The BGE comparison needs extra packages and a first-time model download:

```bash
pip install -r requirements-optional.txt
export HF_ENDPOINT=https://hf-mirror.com
python src/classify_embedding.py --full --text-chars 300 --folds 10 --backend sentence-transformers --device cuda
```

On PowerShell, set `$env:HF_ENDPOINT = "https://hf-mirror.com"`. On CPU, use `--backend fastembed --device cpu`; the full `sentence-transformers` run is not practical without a GPU.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: streamlit` | `pip install -r requirements-dashboard.txt` |
| Missing parquet input | Run the data-preparation steps or `python src/run_all.py --only parse,scope,build_features` |
| Empty boxes in figures | Install a CJK font and rebuild; see [figure reproduction](figure_reproduction.md) |
| Dashboard shows old numbers | Re-run `python src/run_all.py --extras`, then restart Streamlit |

The full Chinese guide, including package versions and platform notes, is [setup](../setup.md).
