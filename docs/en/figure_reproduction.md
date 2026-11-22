# Figure Reproduction

All 44 result figures in `results/figures/` can be regenerated from the repository; external reports or documents are not required.

## Entry Point

```bash
python src/figure_pipeline.py --check
python src/figure_pipeline.py --all
```

- `--check` verifies the fixed set of 44 files: present, non-empty, readable by Pillow, at least 600 × 300 px.
- `--all` redraws the table-driven figures, then checks the full set.
- Missing experiment-only figures produce a hint with the command that creates them.

The complete path, including retraining the main pipeline and supplementary experiments:

```bash
python src/run_all.py --extras
```

`--extras` ends with the `figures` step, which redraws and validates the 44 figures.

## Two Kinds of Figures

**33 table-driven figures** are rendered from committed `results/tables/*.json` and `results/tables/*.csv`; `--all` can redraw them without retraining.

**11 experiment-only figures** are written directly by experiment scripts in their final style:

```text
fig20, fig21, fig23, fig25
fig30, fig31, fig32, fig33
fig43, fig49, fig50
```

They come from `cluster.py`, `classify.py`, `classify_time.py`, `classify_subgroups.py`, and `feature_selection.py`. Run those steps or `python src/run_all.py --extras` before `--all` if they are missing.

## Inputs

Generated inputs use `data/processed/` and `results/tables/`. The repository commits the lightweight result tables and figures, but not parquet files. A fresh clone therefore needs at least:

```bash
python src/run_all.py --only parse,scope,build_features
```

For figures that depend on cluster assignments or decision-curve outputs, run the corresponding experiments:

```bash
python src/cluster.py --force
python src/classify_decision.py
```

`figure_pipeline --all` reports the specific missing input and command when one is unavailable.

## Fonts

Matplotlib searches `Microsoft YaHei → SimHei → Noto Sans CJK SC → PingFang SC`. The pipeline warns when no CJK font is found. Empty glyph boxes mean the font is missing, not that the data is wrong.

## CI

CI runs `python src/figure_pipeline.py --check` on every push/PR to `dev` or `main`. It validates the committed images but does not redraw or retrain. Full redraw remains local, via `python src/run_all.py --extras`.

The Chinese version with the detailed file-to-input mapping is [figure_reproduction](../figure_reproduction.md).
