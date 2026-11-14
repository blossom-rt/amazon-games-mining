"""also_buy / also_view / 共同评论人三张商品网络的对比。

不把 also_buy 当购买篮，只比较三张网络的边结构：
- also_buy：商品页推荐位；
- also_view：浏览关联；
- co-reviewer：同一评论人评论过的商品对（只取 2–5 件商品的普通用户，避免超级用户）。

输出跨叶子占比、Jaccard 重叠、跨品类 top 边。

用法：
    python src/network_compare.py
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from .build_features import category_leaf, clean_main_cat, clean_text
    from .config import DATA_PROCESSED, FIGURES, FONT_CANDIDATES, SEED, TABLES
except ImportError:
    from build_features import category_leaf, clean_main_cat, clean_text
    from config import DATA_PROCESSED, FIGURES, FONT_CANDIDATES, SEED, TABLES

META = DATA_PROCESSED / "meta.parquet"
SCOPE = DATA_PROCESSED / "scope.parquet"
REVIEWS = DATA_PROCESSED / "reviews.parquet"
OUT_JSON = TABLES / "network_compare.json"
OUT_TOP = TABLES / "network_compare_top.csv"
OUT_FIG = FIGURES / "fig46_network_compare.png"
MAX_BASKET = 50
MAX_REVIEWERS = 20000


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False


def edges_from_baskets(baskets: list[list[str]], in_scope: set[str]) -> Counter:
    counter = Counter()
    for items in baskets:
        unique = sorted({item for item in items[:MAX_BASKET] if item in in_scope})
        for left, right in combinations(unique, 2):
            counter[(left, right)] += 1
    return counter


def main() -> None:
    setup_fonts()
    meta = pq.read_table(META, columns=["asin", "title", "also_buy", "also_view", "category", "main_cat"]).to_pandas()
    scope = pq.read_table(SCOPE, columns=["asin", "in_scope"]).to_pandas()
    in_scope = set(scope.loc[scope["in_scope"], "asin"])
    meta = meta.loc[meta["asin"].isin(in_scope)].copy()
    leaf = {}
    for asin, value in zip(meta["asin"], meta["category"]):
        if value is not None and hasattr(value, "__len__"):
            leaf[asin] = category_leaf(list(value))
    broad = dict(zip(meta["asin"], meta["main_cat"].map(clean_main_cat).fillna("(缺失)")))
    title = dict(zip(meta["asin"], meta["title"].map(clean_text)))

    buy = edges_from_baskets(meta["also_buy"].tolist(), in_scope)
    view = edges_from_baskets(meta["also_view"].tolist(), in_scope)

    reviews = pq.read_table(REVIEWS, columns=["reviewerID", "asin"]).to_pandas()
    reviews = reviews.loc[reviews["asin"].isin(in_scope)]
    groups = reviews.groupby("reviewerID")["asin"].apply(lambda values: sorted(set(values)))
    groups = groups.loc[(groups.map(len) >= 2) & (groups.map(len) <= 5)]
    rng = np.random.default_rng(SEED)
    if len(groups) > MAX_REVIEWERS:
        chosen = rng.choice(groups.index.to_numpy(), size=MAX_REVIEWERS, replace=False)
        groups = groups.loc[chosen]
    co = Counter()
    for values in groups:
        for left, right in combinations(values, 2):
            co[(left, right)] += 1

    def cross_share(counter: Counter, mapping: dict) -> float:
        if not counter:
            return 0.0
        cross = sum(1 for left, right in counter if mapping.get(left) != mapping.get(right))
        return cross / len(counter)

    def summary(counter: Counter) -> dict:
        return {
            "edges": int(len(counter)),
            "cross_leaf_share": round(cross_share(counter, leaf), 4),
            "cross_main_share": round(cross_share(counter, broad), 4),
            "multi_edge_share": round(float(np.mean([count >= 2 for count in counter.values()])), 4),
            "median_edge_weight": float(np.median(list(counter.values()))),
        }

    buy_summary, view_summary, co_summary = summary(buy), summary(view), summary(co)
    buy_edges, view_edges, co_edges = set(buy), set(view), set(co)

    def jaccard(left: set, right: set) -> float:
        return round(len(left & right) / len(left | right), 4) if left | right else 0.0

    overlap = {
        "buy_view_jaccard": jaccard(buy_edges, view_edges),
        "buy_co_jaccard": jaccard(buy_edges, co_edges),
        "view_co_jaccard": jaccard(view_edges, co_edges),
        "buy_edges_in_view": round(len(buy_edges & view_edges) / len(buy_edges), 4) if buy_edges else 0.0,
        "buy_edges_in_co": round(len(buy_edges & co_edges) / len(buy_edges), 4) if buy_edges else 0.0,
    }

    rows = []
    for left, right in buy_edges:
        if leaf.get(left) == leaf.get(right):
            continue
        rows.append({
            "network": "also_buy",
            "left": left,
            "right": right,
            "left_title": title.get(left, ""),
            "right_title": title.get(right, ""),
            "left_leaf": leaf.get(left, ""),
            "right_leaf": leaf.get(right, ""),
            "weight": int(buy[(left, right)]),
        })
    cross_buy = pd.DataFrame(rows).sort_values("weight", ascending=False).head(20) if rows else pd.DataFrame()
    cross_buy.to_csv(OUT_TOP, index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    names = ["also_buy", "also_view", "co-reviewer"]
    shares = [buy_summary["cross_main_share"], view_summary["cross_main_share"], co_summary["cross_main_share"]]
    axes[0].bar(names, shares, color=["#4c72b0", "#dd8452", "#55a868"])
    axes[0].set_ylabel("跨主品类边占比")
    axes[0].set_title("三张网络的跨品类比例")
    axes[1].bar(["buy∩view", "buy∩co", "view∩co"], [overlap["buy_view_jaccard"], overlap["buy_co_jaccard"], overlap["view_co_jaccard"]], color="#8172b2")
    axes[1].set_ylabel("Jaccard")
    axes[1].set_title("边重叠")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(figure)

    result = {
        "n_products": int(len(meta)),
        "n_reviewers_sampled": int(len(groups)),
        "also_buy": buy_summary,
        "also_view": view_summary,
        "co_reviewer": co_summary,
        "overlap": overlap,
        "top_cross_buy": cross_buy.to_dict("records") if not cross_buy.empty else [],
        "outputs": {"top_csv": str(OUT_TOP), "figure": str(OUT_FIG)},
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["also_buy", "also_view", "co_reviewer", "overlap", "n_reviewers_sampled"]}, ensure_ascii=False, indent=2))
    print(f"写入 {OUT_JSON}")


if __name__ == "__main__":
    main()
