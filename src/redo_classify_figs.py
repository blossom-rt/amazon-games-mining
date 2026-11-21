"""用已保存的指标与 SHAP 结果重绘分类指标图。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"], "axes.unicode_minus": False})

from classify import plot_metrics, plot_shap

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"


def main() -> None:
    summary = json.loads((TABLES / "classify_summary.json").read_text(encoding="utf-8"))
    plot_metrics(summary["models"], FIGURES / "fig35_classify_metrics.png")

    shap = json.loads((TABLES / "classify_shap.json").read_text(encoding="utf-8"))
    contributions = np.asarray(shap["contributions"], dtype=float)
    plot_shap(
        contributions,
        shap["base_value"],
        shap["feature_names"],
        shap["samples"],
        FIGURES / "fig34_classify_shap.png",
    )


if __name__ == "__main__":
    main()
