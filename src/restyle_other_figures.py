"""重绘未进入报告正文的图，统一去标题、面板标注、字号与配色。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

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

GREEN = "#55A868"
RED = "#C44E52"
BLUE = "#4C72B0"
ORANGE = "#DD8452"
PURPLE = "#8172B3"
GREY = "#8C8C8C"
CLUSTER_COLORS = {
    "中等热度": BLUE,
    "高热度·共购密集": RED,
    "长尾·共购稀疏": "#CCB974",
    "中等热度·Software": "#64B5CD",
}


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)
    print("wrote", name)


def panel(axis, text: str) -> None:
    axis.text(-0.13, 1.06, f"({text})", transform=axis.transAxes, fontsize=9, fontweight="bold")


def clean(axis, grid: str = "y") -> None:
    axis.grid(axis=grid, alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)


def class_balance() -> None:
    reviews = pd.read_parquet(DATA / "reviews.parquet", columns=["overall"])
    counts = [(reviews["overall"] >= 4).sum(), (reviews["overall"] <= 3).sum()]
    labels = ["正向（≥4 星）", "负向（≤3 星）"]
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    bars = ax.bar(labels, counts, color=[GREEN, RED], width=0.6)
    for bar, value in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.02,
            f"{value / sum(counts) * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("评论数")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 10000:.0f} 万"))
    ax.set_ylim(0, max(counts) * 1.12)
    clean(ax)
    save(fig, "fig02_class_balance.png")


def review_length() -> None:
    features = pd.read_parquet(DATA / "features.parquet", columns=["review_len"])
    values = features["review_len"].to_numpy()
    values = values[(values > 0) & (values <= 5000)]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    bins = np.logspace(0, np.log10(5000), 70)
    ax.hist(values, bins=bins, color=GREEN, alpha=0.85)
    ax.set_xscale("log")
    ax.axvline(np.median(values), color=RED, ls="--", lw=1.2, label=f"中位 {int(np.median(values))} 字符")
    ax.axvline(np.percentile(values, 90), color=ORANGE, ls="--", lw=1.2, label=f"p90 {int(np.percentile(values, 90))} 字符")
    ax.set_xlabel("字符数")
    ax.set_ylabel("评论数")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 10000:.0f} 万"))
    ax.legend(frameon=False, loc="upper right")
    clean(ax)
    save(fig, "fig07_review_length.png")


def vote_style() -> None:
    features = pd.read_parquet(DATA / "features.parquet", columns=["overall", "has_style", "has_vote"])
    grouped = features.groupby("overall")[["has_style", "has_vote"]].mean() * 100
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(grouped.index, grouped["has_style"], marker="o", color=BLUE, label="style 非空率")
    ax.plot(grouped.index, grouped["has_vote"], marker="s", color=ORANGE, label="vote 非空率")
    for x, y in grouped["has_vote"].items():
        if x in (1, 5):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, color=ORANGE)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("星级")
    ax.set_ylabel("非空率")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(frameon=False, loc="upper left")
    clean(ax)
    save(fig, "fig10_vote_style_by_rating.png")


def basket_length() -> None:
    features = pd.read_parquet(DATA / "features.parquet", columns=["asin", "also_buy_len"])
    lengths = features.groupby("asin")["also_buy_len"].max().fillna(0).to_numpy()
    lengths = np.clip(lengths[lengths > 0], 1, 100)
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.hist(lengths, bins=np.arange(0, 102, 2), color=BLUE, alpha=0.9)
    ax.axvline(np.median(lengths), color=RED, ls="--", lw=1.2, label=f"中位 {np.median(lengths):.0f}")
    ax.axvline(lengths.mean(), color=ORANGE, ls="--", lw=1.2, label=f"均值 {lengths.mean():.1f}")
    ax.set_xlabel("每件商品的共购数量（上限 100）")
    ax.set_ylabel("商品数")
    ax.legend(frameon=False, loc="upper right")
    clean(ax)
    save(fig, "fig11_basket_length.png")


def lift_distribution() -> None:
    rules = pd.read_csv(TABLES / "rules_top.csv")
    lifts = rules["lift"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.hist(lifts, bins=45, color=BLUE, alpha=0.9)
    ax.axvline(np.median(lifts), color=RED, ls="--", lw=1.2, label=f"中位 {np.median(lifts):.1f}")
    ax.set_xlabel("提升度 lift")
    ax.set_ylabel("规则数")
    ax.legend(frameon=False, loc="upper right")
    clean(ax)
    save(fig, "fig15_lift_distribution.png")


def top_items() -> None:
    meta = pd.read_parquet(DATA / "meta.parquet", columns=["asin", "title", "also_buy"])
    scope = pd.read_parquet(DATA / "scope.parquet")
    in_scope = set(scope.loc[scope["in_scope"], "asin"])
    title = dict(zip(meta["asin"], meta["title"].fillna(meta["asin"])))
    counts: dict[str, int] = {}
    for row in meta.loc[meta["asin"].isin(in_scope), "also_buy"]:
        for item in row:
            if item in in_scope:
                counts[item] = counts.get(item, 0) + 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:20]
    labels = [(str(title.get(asin, asin))[:30] + "…") if len(str(title.get(asin, asin))) > 30 else str(title.get(asin, asin)) for asin, _ in top][::-1]
    values = [value for _, value in top][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.barh(labels, values, color=GREEN)
    ax.tick_params(axis="y", labelsize=6.8)
    ax.set_xlabel("出现在多少个 also_buy 列表中")
    clean(ax, grid="x")
    save(fig, "fig16_top_items.png")


def cluster_pca() -> None:
    assignments = pd.read_parquet(TABLES / "cluster_assignments.parquet")
    features = pd.read_parquet(DATA / "features.parquet")
    profiles = pd.read_csv(TABLES / "cluster_profiles.csv")
    names = dict(zip(profiles["cluster"], profiles["name"]))
    product = features.groupby("asin").agg(
        n_reviews=("overall", "size"),
        log_price=("log_price", "first"),
        mean_review_len=("review_len", "mean"),
        verified_share=("verified", "mean"),
        also_buy_len=("also_buy_len", "first"),
        also_view_len=("also_view_len", "first"),
        feature_count=("feature_count", "first"),
        description_len=("description_len", "first"),
        category_depth=("category_depth", "first"),
        rank_count=("rank_count", "first"),
    ).reset_index()
    columns = [
        "n_reviews", "log_price", "mean_review_len", "verified_share", "also_buy_len",
        "also_view_len", "feature_count", "description_len", "category_depth", "rank_count",
    ]
    merged = product.merge(assignments[["asin", "cluster"]], on="asin", how="inner")
    merged = merged.sample(min(6000, len(merged)), random_state=42)
    matrix = merged[columns].to_numpy(dtype=float)
    matrix = np.nan_to_num(matrix, nan=np.nanmedian(matrix, axis=0), posinf=0, neginf=0)
    # 评论数、描述/共购长度是重尾变量，先 log1p 再标准化，避免第一主成分被少数头部商品支配。
    for column in ("n_reviews", "mean_review_len", "description_len", "also_buy_len", "also_view_len", "feature_count", "rank_count"):
        index = columns.index(column)
        matrix[:, index] = np.log1p(np.clip(matrix[:, index], 0, None))
    matrix = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-9)
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    coords = u[:, :2] * s[:2]
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for cluster, group in merged.groupby("cluster"):
        idx = group.index.map(lambda x: merged.index.get_loc(x)).to_numpy()
        ax.scatter(coords[idx, 0], coords[idx, 1], s=7, alpha=0.65, color=CLUSTER_COLORS.get(names.get(cluster), GREY), label=names.get(cluster, str(cluster)))
    ax.set_xlabel("主成分 1")
    ax.set_ylabel("主成分 2")
    ax.legend(frameon=False, markerscale=2, loc="best")
    clean(ax, grid="both")
    save(fig, "fig22_clusters_pca.png")


def cluster_sizes() -> None:
    profiles = pd.read_csv(TABLES / "cluster_profiles.csv")
    labels = profiles["name"].tolist()
    values = profiles["size"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    positions = np.arange(len(labels))
    bars = ax.bar(positions, values, color=[CLUSTER_COLORS.get(name, GREY) for name in labels])
    for bar, value, share in zip(bars, values, profiles["share"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.015, f"{value:,}\n({share * 100:.1f}%)", ha="center", va="bottom", fontsize=6.5)
    ax.set_ylabel("商品数")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylim(0, max(values) * 1.15)
    clean(ax)
    save(fig, "fig24_cluster_sizes.png")


def price_reviews() -> None:
    assignments = pd.read_parquet(TABLES / "cluster_assignments.parquet")
    features = pd.read_parquet(DATA / "features.parquet")
    product = features.groupby("asin").agg(price=("price", "first"), reviews=("asin", "size")).reset_index()
    merged = product.merge(assignments[["asin", "cluster"]], on="asin", how="inner")
    profiles = pd.read_csv(TABLES / "cluster_profiles.csv")
    names = dict(zip(profiles["cluster"], profiles["name"]))
    merged = merged[(merged["price"] > 0) & (merged["reviews"] > 0)]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for cluster, group in merged.groupby("cluster"):
        ax.scatter(group["price"], group["reviews"], s=5, alpha=0.4, color=CLUSTER_COLORS.get(names.get(cluster), GREY), label=names.get(cluster, str(cluster)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("价格（美元，对数轴）")
    ax.set_ylabel("评论数（对数轴）")
    ax.legend(frameon=False, markerscale=2, loc="upper right")
    clean(ax, grid="both")
    save(fig, "fig26_price_reviews.png")


def calibration() -> None:
    oof = pd.read_parquet(TABLES / "classify_decision_oof.parquet")
    y = oof["label"].to_numpy()
    raw = oof["probability_raw"].to_numpy()
    cal = oof["probability_calibrated"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for probs, color, label in ((raw, BLUE, "原始概率"), (cal, ORANGE, "isotonic 校准")):
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(probs, bins) - 1
        xs, ys = [], []
        for i in range(10):
            mask = idx == i
            if mask.sum() >= 50:
                xs.append(probs[mask].mean())
                ys.append(y[mask].mean())
        axes[0].plot(xs, ys, marker="o", ms=3, color=color, label=label)
    axes[0].plot([0, 1], [0, 1], ls="--", color=GREY, lw=1, label="理想校准")
    axes[0].set_xlabel("预测概率")
    axes[0].set_ylabel("实际正类比例")
    axes[0].legend(frameon=False, loc="upper left")
    clean(axes[0], grid="both")
    axes[1].hist(raw, bins=40, alpha=0.45, color=BLUE, label="原始")
    axes[1].hist(cal, bins=40, alpha=0.45, color=ORANGE, label="校准后")
    axes[1].set_xlabel("预测概率")
    axes[1].set_ylabel("样本数")
    axes[1].legend(frameon=False, loc="upper left")
    clean(axes[1])
    panel(axes[0], "a")
    panel(axes[1], "b")
    save(fig, "fig36_calibration.png")


def threshold() -> None:
    oof = pd.read_parquet(TABLES / "classify_decision_oof.parquet")
    y = oof["label"].to_numpy()
    probs = oof["probability_calibrated"].to_numpy()
    grid = np.linspace(0.05, 0.90, 120)
    f1 = []
    for t in grid:
        pred = probs >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        fn = np.sum(~pred & (y == 1))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1.append(2 * precision * recall / (precision + recall + 1e-9))
    best = grid[int(np.argmax(f1))]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(grid, f1, color=BLUE)
    ax.axvline(best, color=GREEN, ls="--", lw=1.1, label=f"F1 最优 {best:.2f}")
    ax.axvline(0.17, color=RED, ls="--", lw=1.1, label="代价敏感 0.17")
    ax.set_xlabel("阈值")
    ax.set_ylabel("宏 F1")
    ax.legend(frameon=False, loc="lower right")
    clean(ax)
    save(fig, "fig37_threshold.png")


def cluster_methods() -> None:
    metrics = pd.read_csv(TABLES / "cluster_compare_metrics.csv")
    profiles = pd.read_csv(TABLES / "cluster_compare_profiles.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    subset = metrics.dropna(subset=["silhouette"])
    axes[0].bar(subset["method"], subset["silhouette"], color=BLUE)
    axes[0].set_ylabel("轮廓系数")
    axes[0].tick_params(axis="x", rotation=15)
    clean(axes[0])
    for method, group in profiles.groupby("method"):
        if method == "hdbscan":
            continue
        group = group.sort_values("cluster")
        axes[1].plot(group["cluster"], group["price_median_usd"], marker="o", ms=4, label=method)
    axes[1].set_xlabel("簇编号")
    axes[1].set_ylabel("价格中位数（有观测价格）")
    axes[1].set_xticks(sorted(profiles["cluster"].unique()))
    axes[1].legend(frameon=False)
    clean(axes[1])
    panel(axes[0], "a")
    panel(axes[1], "b")
    save(fig, "fig39_cluster_methods.png")


def cluster_validation() -> None:
    outcomes = pd.read_csv(TABLES / "cluster_validation_outcomes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    axes[0].bar(outcomes["cluster"].astype(str), outcomes["mean_overall"], color=BLUE)
    axes[0].set_xlabel("簇编号")
    axes[0].set_ylabel("留出集平均评分")
    axes[0].set_ylim(0, 5)
    clean(axes[0])
    axes[1].bar(outcomes["cluster"].astype(str), outcomes["mean_n_reviews"], color=ORANGE)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("簇编号")
    axes[1].set_ylabel("留出集平均评论数（对数轴）")
    clean(axes[1])
    panel(axes[0], "a")
    panel(axes[1], "b")
    save(fig, "fig45_cluster_validation.png")


def main() -> None:
    class_balance()
    review_length()
    vote_style()
    basket_length()
    lift_distribution()
    top_items()
    cluster_pca()
    cluster_sizes()
    price_reviews()
    calibration()
    threshold()
    cluster_methods()
    cluster_validation()


if __name__ == "__main__":
    main()
