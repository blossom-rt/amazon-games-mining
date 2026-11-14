"""聚类与业务结果联动：四个簇的评分、评论量、verified 与价格观测差异。

聚类本身不使用评分；这里只做事后检验，回答“哪个簇值得运营”。
输出：
    results/tables/cluster_outcomes.csv
    results/tables/cluster_outcomes.json
    results/figures/fig42_cluster_outcomes.png

用法：
    python src/cluster_insights.py
"""

from __future__ import annotations

import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import kruskal

try:
    from .config import FEATURES_PARQUET, FIGURES, FONT_CANDIDATES, SEED, TABLES
except ImportError:
    from config import FEATURES_PARQUET, FIGURES, FONT_CANDIDATES, SEED, TABLES

ASSIGNMENTS = TABLES / "cluster_assignments.parquet"
OUT_CSV = TABLES / "cluster_outcomes.csv"
OUT_JSON = TABLES / "cluster_outcomes.json"
OUT_FIG = FIGURES / "fig42_cluster_outcomes.png"
METRICS = [
    "mean_overall",
    "n_reviews",
    "mean_review_len",
    "verified_share",
    "price_observed",
    "mean_also_buy",
]


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False


def epsilon_squared(statistic: float, n: int, groups: int) -> float:
    return float((statistic - groups + 1) / (n - groups)) if n > groups else 0.0


def product_frame() -> pd.DataFrame:
    frame = pq.read_table(
        FEATURES_PARQUET,
        columns=["asin", "in_scope", "overall", "review_len", "verified", "price", "also_buy_len"],
    ).to_pandas()
    frame = frame.loc[frame["in_scope"]]
    grouped = frame.groupby("asin").agg(
        mean_overall=("overall", "mean"),
        n_reviews=("overall", "size"),
        mean_review_len=("review_len", "mean"),
        verified_share=("verified", "mean"),
        price_observed=("price", lambda values: float(values.notna().mean())),
        mean_also_buy=("also_buy_len", "mean"),
    ).reset_index()
    assignments = pq.read_table(ASSIGNMENTS).to_pandas()
    merged = assignments.merge(grouped, on="asin", how="left", validate="one_to_one")
    assert len(merged) == len(assignments), "簇分配与商品聚合不完整"
    merged["n_reviews"] = merged["n_reviews"].fillna(0).astype(int)
    return merged


def plot_outcomes(frame: pd.DataFrame, path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    groups = [frame.loc[frame["cluster"] == cluster, "mean_overall"].dropna().to_numpy() for cluster in sorted(frame["cluster"].unique())]
    positions = np.arange(len(groups))
    axes[0].boxplot(groups, positions=positions, tick_labels=[f"簇 {i}" for i in range(len(groups))], showfliers=False)
    axes[0].set_ylabel("商品平均评分")
    axes[0].set_title("各簇平均评分分布")
    axes[0].grid(axis="y", alpha=0.25)

    groups = [frame.loc[frame["cluster"] == cluster, "n_reviews"].to_numpy() for cluster in sorted(frame["cluster"].unique())]
    positions = np.arange(len(groups))
    axes[1].boxplot(groups, positions=positions, tick_labels=[f"簇 {i}" for i in range(len(groups))], showfliers=False)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("商品评论数（对数）")
    axes[1].set_title("各簇评论量分布")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    setup_fonts()
    frame = product_frame()
    clusters = sorted(frame["cluster"].unique())
    rows = []
    tests = {}
    for cluster in clusters:
        part = frame.loc[frame["cluster"] == cluster]
        row = {
            "cluster": int(cluster),
            "n_products": int(len(part)),
            "mean_overall": round(float(part["mean_overall"].mean()), 4),
            "n_reviews": round(float(part["n_reviews"].mean()), 1),
            "mean_review_len": round(float(part["mean_review_len"].mean()), 1),
            "verified_share": round(float(part["verified_share"].mean()), 4),
            "price_observed": round(float(part["price_observed"].mean()), 4),
            "mean_also_buy": round(float(part["mean_also_buy"].mean()), 1),
        }
        rows.append(row)

    for metric in METRICS:
        groups = [
            frame.loc[frame["cluster"] == cluster, metric].dropna().to_numpy()
            for cluster in clusters
        ]
        statistic, p_value = kruskal(*groups)
        tests[metric] = {
            "statistic": round(float(statistic), 2),
            "p_value": float(p_value),
            "epsilon_squared": round(epsilon_squared(float(statistic), len(frame), len(clusters)), 4),
        }

    outcomes = pd.DataFrame(rows)
    outcomes.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    plot_outcomes(frame, OUT_FIG)

    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED,
        "n_products": int(len(frame)),
        "clusters": outcomes.to_dict("records"),
        "kruskal": tests,
        "notes": "聚类未使用评分；评分、verified、价格只在事后检验。",
        "outputs": {"csv": str(OUT_CSV), "figure": str(OUT_FIG)},
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(outcomes.to_string(index=False))
    print(json.dumps(tests, ensure_ascii=False, indent=2))
    print(f"写入 {OUT_JSON}")


if __name__ == "__main__":
    main()
