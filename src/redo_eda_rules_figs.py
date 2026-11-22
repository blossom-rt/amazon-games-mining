"""重绘 EDA 与规则图，去掉图内标题并统一小字号。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
FEATURES = ROOT / "data" / "processed" / "features.parquet"
META = ROOT / "data" / "processed" / "meta.parquet"

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"],
        "axes.unicode_minus": False,
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 200,
    }
)
BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"
GREY = "#8C8C8C"


def panel(ax, tag: str) -> None:
    ax.text(-0.12, 1.06, f"({tag})", transform=ax.transAxes, fontsize=9, fontweight="bold")


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)
    print("wrote", name)


def main() -> None:
    features = pq.read_table(
        FEATURES, columns=["asin", "review_year", "main_cat_clean", "overall"]
    ).to_pandas()
    meta = pq.read_table(META, columns=["asin", "main_cat", "also_buy", "price"]).to_pandas()

    yearly = features.groupby("review_year").size()
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    ax.bar(yearly.index, yearly.values, color=BLUE, width=0.8)
    peak = int(yearly.idxmax())
    ax.annotate(
        f"{peak} 年\n{yearly.max():,} 条",
        xy=(peak, yearly.max()),
        xytext=(peak - 4, yearly.max() * 0.88),
        fontsize=7,
        arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 0.8},
    )
    ax.set_xlabel("评论年份")
    ax.set_ylabel("评论数")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/10000:.0f} 万"))
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig03_reviews_over_time.png")

    category = features["main_cat_clean"].fillna("(缺失)").replace("", "(缺失)")
    counts = features.assign(category=category).groupby("category")["asin"].nunique().sort_values(ascending=False).head(10)
    counts = counts.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bars = ax.barh(counts.index, counts.values, color=[GREY if i == "(缺失)" else BLUE for i in counts.index])
    for bar, value in zip(bars, counts.values):
        ax.text(value + counts.max() * 0.01, bar.get_y() + bar.get_height() / 2, f"{value:,}", va="center", fontsize=7)
    ax.set_xlabel("商品数")
    ax.set_xlim(0, counts.max() * 1.12)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig04_main_cat.png")

    per_product = features.groupby("asin").size()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bins = np.logspace(0, np.log10(per_product.max() + 1), 40)
    ax.hist(per_product, bins=bins, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("单件商品评论数")
    ax.set_ylabel("商品数")
    ax.axvline(per_product.median(), color=RED, linestyle="--", linewidth=1, label=f"中位 {per_product.median():.0f}")
    ax.legend()
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig09_reviews_per_product.png")

    basket = meta["also_buy"].apply(lambda v: len(v) if isinstance(v, (list, np.ndarray)) else 0)
    basket = basket[basket > 0]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    axes[0].hist(basket, bins=50, color=BLUE, edgecolor="white", linewidth=0.3)
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("每件商品的共购数量")
    axes[0].set_ylabel("商品数")
    sorted_basket = np.sort(basket.values)
    cumulative = np.arange(1, len(sorted_basket) + 1) / len(sorted_basket)
    axes[1].plot(sorted_basket, cumulative, color=GREEN)
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 1.01)
    axes[1].set_xlabel("每件商品的共购数量")
    axes[1].set_ylabel("累计占比")
    for ax, tag in zip(axes, "ab"):
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        panel(ax, tag)
    save(fig, "fig05_basket_length.png")

    price = pd.to_numeric(meta["price"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    positive = price[price > 0]
    axes[0].hist(positive, bins=np.logspace(np.log10(positive.min()), np.log10(positive.max()), 60), color=BLUE)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("价格（美元，对数轴）")
    axes[0].set_ylabel("商品数")
    axes[0].axvline(positive.median(), color=RED, linestyle="--", linewidth=1, label=f"中位 ${positive.median():.2f}")
    axes[0].legend()
    main = meta[["asin", "price"]].merge(
        features[["asin", "main_cat_clean"]].drop_duplicates("asin"), on="asin", how="left"
    )
    main["category"] = main["main_cat_clean"].fillna("(缺失)")
    grouped = main.groupby("category")["price"].agg(["size", "count"])
    grouped["missing_rate"] = 1 - grouped["count"] / grouped["size"]
    grouped = grouped[grouped["size"] >= 100].sort_values("missing_rate", ascending=False).head(8)
    grouped = grouped.iloc[::-1]
    axes[1].barh(grouped.index, grouped["missing_rate"] * 100, color=RED)
    axes[1].set_xlabel("价格缺失率（%）")
    for ax, tag in zip(axes, "ab"):
        ax.grid(axis="x" if ax is axes[1] else "y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        panel(ax, tag)
    save(fig, "fig06_price_distribution.png")

    rules = pd.read_csv(TABLES / "rules_all.csv")
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="viridis", s=8, alpha=0.7)
    ax.set_xlabel("支持度 support")
    ax.set_ylabel("置信度 confidence")
    colorbar = fig.colorbar(ax.collections[0], ax=ax)
    colorbar.set_label("提升度 lift")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig12_rule_scatter.png")

    runtime = pd.read_csv(TABLES / "rules_runtime.csv")
    pairs = runtime[["min_support", "min_confidence"]].drop_duplicates().values
    labels = [f"{s:.3f}/{c:.2f}" for s, c in pairs]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    apriori = runtime[runtime["algorithm"] == "apriori"]["seconds"].values
    fpgrowth = runtime[runtime["algorithm"] == "fpgrowth"]["seconds"].values
    ax.bar(x - width / 2, apriori, width, label="Apriori", color=BLUE)
    ax.bar(x + width / 2, fpgrowth, width, label="FP-Growth", color=ORANGE)
    ax.set_xticks(x, labels)
    ax.set_yscale("log")
    ax.set_xlabel("support / confidence")
    ax.set_ylabel("耗时（秒，对数轴）")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig14_runtime.png")


if __name__ == "__main__":
    main()
