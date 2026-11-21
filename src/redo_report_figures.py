"""重绘报告补充实验图，统一去标题、配色与字号。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
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


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)
    print("wrote", name)


def panel(axis, text: str) -> None:
    axis.text(-0.12, 1.06, f"({text})", transform=axis.transAxes, fontsize=9, fontweight="bold")


def redo_rating() -> None:
    labels = ["1 星", "2 星", "3 星", "4 星", "5 星"]
    reviews = pd.read_parquet(ROOT / "data" / "processed" / "reviews.parquet", columns=["overall"])
    counts = reviews["overall"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0).to_numpy()
    shares = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    bars = ax.bar(labels, counts, color=[RED, "#DD8452", "#CCB974", GREEN, "#3B7A57"])
    for bar, share in zip(bars, shares):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.015,
            f"{share:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    ax.set_ylabel("评论数")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/10000:.0f} 万"))
    ax.set_ylim(0, counts.max() * 1.12)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig01_rating_distribution.png")


def redo_missing() -> None:
    rows = [
        ("商品.tech1", 96.1),
        ("商品.price", 83.4),
        ("评论.vote", 82.7),
        ("商品.also_buy 空列表", 60.1),
        ("评论.style", 48.5),
        ("商品.description", 12.5),
        ("商品.brand", 5.0),
        ("商品.category 空列表", 2.9),
        ("商品.main_cat", 0.2),
        ("评论.reviewText", 0.1),
    ]
    rows = rows[::-1]
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    bars = ax.barh(names, values, color=[GREY if v > 20 else BLUE for v in values])
    for bar, value in zip(bars, values):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=7)
    ax.set_xlim(0, 105)
    ax.set_xlabel("缺失率（%）")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig08_missing_rates.png")


def redo_rules() -> None:
    top = pd.read_csv(TABLES / "rules_top.csv").head(15).iloc[::-1]
    labels = []
    for _, row in top.iterrows():
        left = str(row["antecedent_titles"]).split(";")[0].split(" - ")[0].strip()
        right = str(row["consequent_title"]).split(" - ")[0].strip()
        labels.append(f"{left[:24]} → {right[:24]}")
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.barh(labels, top["lift"], color=[PURPLE if bool(s) else BLUE for s in top["same_series"]])
    for y, value in enumerate(top["lift"]):
        ax.text(value + 1.5, y, f"{value:.0f}", va="center", fontsize=7)
    ax.set_xlim(0, top["lift"].max() * 1.13)
    ax.set_xlabel("提升度 lift")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig13_top_lift.png")


def redo_embedding() -> None:
    data = pd.read_csv(TABLES / "classify_embedding_metrics.csv")
    metrics = ["macro_f1", "pr_auc", "roc_auc"]
    names = ["宏 F1", "PR-AUC", "ROC-AUC"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.52))
    for ax, metric, name, tag in zip(axes, metrics, names, "abc"):
        summary = data.groupby("representation")[metric].agg(["mean", "std"]).loc[["bge_small", "hash_tfidf"]]
        x = np.arange(2)
        ax.bar(x, summary["mean"], yerr=summary["std"], capsize=3, color=[BLUE, ORANGE], width=0.58)
        ax.set_xticks(x, ["BGE", "TF-IDF"])
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel(name)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        panel(ax, tag)
    save(fig, "fig38_embedding_compare.png")


def redo_ablation() -> None:
    data = pd.read_csv(TABLES / "classify_ablation_metrics.csv")
    models = ["text_only", "metadata_only", "combined"]
    labels = ["仅文本", "仅元数据", "文本+元数据"]
    colors = [BLUE, ORANGE, GREEN]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, metric, name, tag in zip(axes, ["macro_f1", "pr_auc"], ["宏 F1", "PR-AUC"], "ab"):
        summary = data.groupby("model")[metric].agg(["mean", "std"]).loc[models]
        ax.bar(labels, summary["mean"], yerr=summary["std"], capsize=3, color=colors, width=0.58)
        ax.set_ylim(0.4 if metric == "macro_f1" else 0.75, 1.0)
        ax.set_ylabel(name)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        panel(ax, tag)
    save(fig, "fig40_ablation.png")


def redo_terms() -> None:
    data = pd.read_csv(TABLES / "classify_terms.csv")
    pos = data[data["direction"] == "positive"].nlargest(12, "coefficient").iloc[::-1]
    neg = data[data["direction"] == "negative"].nsmallest(12, "coefficient").iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, frame, color, tag in zip(axes, [neg, pos], [RED, GREEN], "ab"):
        ax.barh(frame["term"], frame["coefficient"], color=color)
        ax.axvline(0, color="#444444", linewidth=0.6)
        ax.set_xlabel("逻辑回归系数")
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        panel(ax, tag)
    save(fig, "fig41_top_terms.png")


def redo_cluster_outcomes() -> None:
    data = pd.read_csv(TABLES / "cluster_outcomes.csv")
    labels = [f"C{int(c)}" for c in data["cluster"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].bar(labels, data["mean_overall"], color=PURPLE, width=0.6)
    axes[0].set_ylim(3.0, 4.2)
    axes[0].set_ylabel("平均评分")
    axes[1].bar(labels, data["n_reviews"], color=ORANGE, width=0.6)
    axes[1].set_ylabel("平均评论数（对数）")
    axes[1].set_yscale("log")
    for ax, tag in zip(axes, "ab"):
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        panel(ax, tag)
    save(fig, "fig42_cluster_outcomes.png")


def redo_negation() -> None:
    data = json.loads((TABLES / "classify_negation.json").read_text(encoding="utf-8"))["variants"]
    names = ["原始", "neg_scope"]
    keys = ["raw", "neg_scope"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axes[0].bar(names, [data[k]["macro_f1"]["mean"] for k in keys], color=[BLUE, GREEN], width=0.55)
    axes[0].set_ylim(0.80, 0.90)
    axes[0].set_ylabel("宏 F1")
    x = np.arange(2)
    width = 0.36
    axes[1].bar(x - width / 2, [data[k]["error_rate_negation"] for k in keys], width, label="含否定词", color=RED)
    axes[1].bar(x + width / 2, [data[k]["error_rate_concession"] for k in keys], width, label="含转折词", color=PURPLE)
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("错误率")
    axes[1].legend()
    for ax, tag in zip(axes, "ab"):
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        panel(ax, tag)
    save(fig, "fig44_negation_scope.png")


def redo_network() -> None:
    data = json.loads((TABLES / "network_compare.json").read_text(encoding="utf-8"))
    labels = ["also_buy", "also_view", "共同评论人"]
    keys = ["also_buy", "also_view", "co_reviewer"]
    shares = [data[k]["cross_main_share"] for k in keys]
    overlap = data["overlap"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].bar(labels, shares, color=[BLUE, ORANGE, GREEN], width=0.58)
    axes[0].set_ylabel("跨主品类边占比")
    axes[0].set_ylim(0, max(shares) * 1.2)
    pairs = ["buy∩view", "buy∩co", "view∩co"]
    values = [overlap["buy_view_jaccard"], overlap["buy_co_jaccard"], overlap["view_co_jaccard"]]
    axes[1].bar(pairs, values, color=PURPLE, width=0.58)
    axes[1].set_ylabel("Jaccard")
    for ax, tag in zip(axes, "ab"):
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        panel(ax, tag)
    save(fig, "fig46_network_compare.png")


def redo_link() -> None:
    data = pd.read_csv(TABLES / "link_prediction_metrics.csv").sort_values("recall@10", ascending=False)
    labels = data["method_label"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    x = np.arange(len(data))
    width = 0.38
    axes[0].bar(x - width / 2, data["recall@10"], width, label="Recall@10", color=BLUE)
    axes[0].bar(x + width / 2, data["recall@50"], width, label="Recall@50", color=ORANGE)
    axes[0].set_ylabel("召回率")
    axes[0].legend()
    axes[1].bar(x - width / 2, data["ndcg@10"], width, label="NDCG@10", color=GREEN)
    axes[1].bar(x + width / 2, data["map@10"], width, label="MAP@10", color=RED)
    axes[1].set_ylabel("排序指标")
    axes[1].legend()
    for ax, tag in zip(axes, "ab"):
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        panel(ax, tag)
    save(fig, "fig47_link_prediction.png")


def redo_decision() -> None:
    data = pd.read_csv(TABLES / "classify_decision_curve.csv")
    summary = json.loads((TABLES / "classify_decision.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.0, 2.63))
    ax.plot(data["threshold"], data["net_benefit_calibrated"], label="isotonic 校准", color=GREEN)
    ax.plot(data["threshold"], data["net_benefit_raw"], label="原始概率", color=BLUE)
    ax.plot(data["threshold"], data["net_benefit_treat_all"], label="全部判正", color=GREY, linestyle="--")
    for key, label, color in [
        ("best_net_benefit_threshold", "净收益最优", RED),
        ("cost_sensitive_fn5", "FN=5×FP", ORANGE),
    ]:
        value = summary[key]["threshold"] if isinstance(summary[key], dict) else summary[key]
        ax.axvline(value, color=color, linestyle=":", linewidth=1.2, label=f"{label} {value:.2f}")
    ax.set_xlabel("判定阈值")
    ax.set_ylabel("净收益")
    ax.set_ylim(0, 0.82)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig48_decision_curve.png")


def main() -> None:
    redo_rating()
    redo_missing()
    redo_rules()
    redo_embedding()
    redo_ablation()
    redo_terms()
    redo_cluster_outcomes()
    redo_negation()
    redo_network()
    redo_link()
    redo_decision()


if __name__ == "__main__":
    main()
