"""also_buy 推荐位网络的链接预测评估。

把 also_buy 看成有向推荐列表，而不是购买篮：

- 对度数足够的商品，留出 20% 的 also_buy 目标作为测试边；
- 用剩余边构造无向共现图，比较热度、共同邻居、Jaccard、Adamic-Adar、
  余弦相似度和 also_view 余弦；
- 报告 Recall@10/50、NDCG@10、MAP@10、覆盖率和跨主品类召回。

用法：
    python src/link_prediction.py
    python src/link_prediction.py --sources 1000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

try:
    from .build_features import clean_main_cat
    from .config import DATA_PROCESSED, FIGURES, FONT_CANDIDATES, SEED, TABLES
except ImportError:
    from build_features import clean_main_cat
    from config import DATA_PROCESSED, FIGURES, FONT_CANDIDATES, SEED, TABLES

META = DATA_PROCESSED / "meta.parquet"
SCOPE = DATA_PROCESSED / "scope.parquet"
OUT_JSON = TABLES / "link_prediction.json"
OUT_METRICS = TABLES / "link_prediction_metrics.csv"
OUT_FIG = FIGURES / "fig47_link_prediction.png"

METHOD_LABELS = {
    "popularity": "热度",
    "common_neighbors": "共同邻居",
    "jaccard": "Jaccard",
    "adamic_adar": "Adamic-Adar",
    "cosine": "余弦",
    "view_cosine": "also_view 余弦",
}


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False


def clean_targets(values, in_scope: set[str], source: str) -> list[str]:
    if values is None or not hasattr(values, "__len__"):
        return []
    result = []
    for value in list(values)[:100]:
        target = str(value)
        if target != source and target in in_scope:
            result.append(target)
    return sorted(set(result))


def build_graph(pairs: set[tuple[str, str]], index: dict[str, int]) -> sp.csr_matrix:
    rows, cols = [], []
    for left, right in pairs:
        i, j = index.get(left), index.get(right)
        if i is None or j is None:
            continue
        if i == j:
            continue
        rows.extend([i, j])
        cols.extend([j, i])
    values = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((values, (rows, cols)), shape=(len(index), len(index)))


def evaluate_scores(
    scores: np.ndarray,
    source_index: list[int],
    train_neighbors: dict[int, set[int]],
    test_targets: dict[int, list[int]],
    item_main: list[str],
    k_max: int = 50,
) -> dict:
    recall10, recall50, ndcg10, ap10 = [], [], [], []
    cross_hits, cross_total = 0, 0
    same_hits, same_total = 0, 0
    top10_items: set[int] = set()
    per_source_hits = []

    for row, source in enumerate(source_index):
        vector = scores[row].copy()
        blocked = set(train_neighbors.get(source, set()))
        blocked.add(source)
        if blocked:
            vector[list(blocked)] = -np.inf

        k = min(k_max, len(vector))
        top = np.argpartition(-vector, k - 1)[:k]
        top = top[np.argsort(-vector[top])]
        rank = {int(item): position + 1 for position, item in enumerate(top)}
        top10_items.update(int(item) for item in top[:10])

        positives = test_targets[source]
        hits = [rank[item] for item in positives if item in rank and rank[item] <= 50]
        hits10 = [value for value in hits if value <= 10]
        recall10.append(len(hits10) / len(positives))
        recall50.append(len(hits) / len(positives))

        dcg = sum(1.0 / math.log2(value + 1) for value in hits10)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(positives), 10) + 1))
        ndcg10.append(dcg / idcg if idcg else 0.0)

        ap = 0.0
        for position, item_rank in enumerate(sorted(hits10), start=1):
            ap += position / item_rank
        ap10.append(ap / min(len(positives), 10) if positives else 0.0)
        per_source_hits.append(len(hits10) / len(positives))

        source_main = item_main[source]
        for target in positives:
            related = item_main[target] == source_main
            if related:
                same_total += 1
                same_hits += int(rank.get(target, 999) <= 10)
            else:
                cross_total += 1
                cross_hits += int(rank.get(target, 999) <= 10)

    return {
        "recall@10": float(np.mean(recall10)),
        "recall@50": float(np.mean(recall50)),
        "ndcg@10": float(np.mean(ndcg10)),
        "map@10": float(np.mean(ap10)),
        "coverage@10": len(top10_items) / len(item_main),
        "cross_recall@10": cross_hits / cross_total if cross_total else None,
        "same_recall@10": same_hits / same_total if same_total else None,
        "per_source_recall10": per_source_hits,
        "cross_positives": cross_total,
        "same_positives": same_total,
    }


def bootstrap_ci(values: list[float], rng: np.random.Generator, n_boot: int = 1000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(array, size=(n_boot, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser(description="also_buy 链接预测")
    parser.add_argument("--sources", type=int, default=2000, help="参与评估的源商品数")
    parser.add_argument("--min-degree", type=int, default=10, help="源商品最小 also_buy 度数")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    setup_fonts()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()

    meta = pq.read_table(
        META,
        columns=["asin", "main_cat", "also_buy", "also_view"],
    ).to_pandas()
    scope = pq.read_table(SCOPE, columns=["asin", "in_scope"]).to_pandas()
    in_scope = set(scope.loc[scope["in_scope"], "asin"])
    meta = meta.loc[meta["asin"].isin(in_scope)].reset_index(drop=True)

    asins = meta["asin"].astype(str).tolist()
    index = {asin: position for position, asin in enumerate(asins)}
    main_cat = [clean_main_cat(value) or "(缺失)" for value in meta["main_cat"].tolist()]

    targets: dict[str, list[str]] = {}
    views: dict[str, list[str]] = {}
    for asin, buy_values, view_values in zip(asins, meta["also_buy"].tolist(), meta["also_view"].tolist()):
        targets[asin] = clean_targets(buy_values, in_scope, asin)
        views[asin] = clean_targets(view_values, in_scope, asin)

    candidates = [asin for asin in asins if len(targets[asin]) >= args.min_degree]
    rng.shuffle(candidates)
    selected = candidates[: args.sources]
    if not selected:
        raise SystemExit("没有满足最小度数的源商品")

    test_targets_by_asin: dict[str, list[str]] = {}
    for source in selected:
        values = targets[source]
        test_size = max(1, int(round(len(values) * 0.2)))
        test = rng.choice(values, size=test_size, replace=False).tolist()
        test_targets_by_asin[source] = test

    test_pairs = {(source, target) for source, values in test_targets_by_asin.items() for target in values}
    train_pairs: set[tuple[str, str]] = set()
    for source, values in targets.items():
        hidden = set(test_targets_by_asin.get(source, []))
        for target in values:
            if target not in hidden:
                train_pairs.add((source, target))

    view_pairs: set[tuple[str, str]] = set()
    for source, values in views.items():
        for target in values:
            view_pairs.add((source, target))

    adjacency = build_graph(train_pairs, index)
    view_adjacency = build_graph(view_pairs, index)
    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float32)
    view_degree = np.asarray(view_adjacency.sum(axis=1)).ravel().astype(np.float32)

    weights = np.zeros_like(degree, dtype=np.float32)
    positive = degree > 1
    weights[positive] = 1.0 / np.log(degree[positive])
    weighted_adjacency = adjacency.multiply(weights[np.newaxis, :])

    source_asin = selected
    source_index = [index[asin] for asin in source_asin]
    train_neighbors = {
        index[source]: {index[target] for target in targets[source] if target not in set(test_targets_by_asin[source])}
        for source in source_asin
    }
    test_targets = {
        index[source]: [index[target] for target in values]
        for source, values in test_targets_by_asin.items()
    }

    results: dict[str, dict] = {}
    batch_size = 128
    for offset in range(0, len(source_index), batch_size):
        batch = source_index[offset : offset + batch_size]
        print(f"评估源商品 {offset + 1}–{offset + len(batch)} / {len(source_index)}")
        rows = adjacency[batch]
        common = (rows @ adjacency).toarray().astype(np.float32)
        view_common = (view_adjacency[batch] @ view_adjacency).toarray().astype(np.float32)
        aa = (rows @ weighted_adjacency).toarray().astype(np.float32)
        degree_source = degree[batch][:, None]
        degree_all = degree[np.newaxis, :]
        view_degree_source = view_degree[batch][:, None]
        view_degree_all = view_degree[np.newaxis, :]

        with np.errstate(divide="ignore", invalid="ignore"):
            union = degree_source + degree_all - common
            jaccard = np.divide(common, union, out=np.zeros_like(common), where=union > 0)
            cosine = np.divide(
                common,
                np.sqrt(degree_source * degree_all),
                out=np.zeros_like(common),
                where=(degree_source * degree_all) > 0,
            )
            view_cosine = np.divide(
                view_common,
                np.sqrt(view_degree_source * view_degree_all),
                out=np.zeros_like(view_common),
                where=(view_degree_source * view_degree_all) > 0,
            )
        popularity = np.broadcast_to(degree[np.newaxis, :], common.shape).copy()

        batch_scores = {
            "popularity": popularity,
            "common_neighbors": common,
            "jaccard": jaccard,
            "adamic_adar": aa,
            "cosine": cosine,
            "view_cosine": view_cosine,
        }
        batch_sources = source_index[offset : offset + len(batch)]
        for method, scores in batch_scores.items():
            metrics = evaluate_scores(scores, batch_sources, train_neighbors, test_targets, main_cat)
            results.setdefault(method, {"rows": []})["rows"].append(metrics)

    rows = []
    for method in METHOD_LABELS:
        chunks = results[method]["rows"]
        merged = {
            key: float(np.mean([chunk[key] for chunk in chunks]))
            for key in ["recall@10", "recall@50", "ndcg@10", "map@10", "coverage@10"]
        }
        cross_values = [chunk["cross_recall@10"] for chunk in chunks if chunk["cross_recall@10"] is not None]
        same_values = [chunk["same_recall@10"] for chunk in chunks if chunk["same_recall@10"] is not None]
        merged["cross_recall@10"] = float(np.mean(cross_values)) if cross_values else None
        merged["same_recall@10"] = float(np.mean(same_values)) if same_values else None
        per_source = np.concatenate([chunk["per_source_recall10"] for chunk in chunks]).tolist()
        merged["recall@10_ci95"] = bootstrap_ci(per_source, rng)
        merged["method"] = method
        merged["method_label"] = METHOD_LABELS[method]
        rows.append(merged)

    metrics_frame = pd.DataFrame(rows).sort_values("recall@10", ascending=False)
    metrics_frame.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")

    best = metrics_frame.iloc[0]["method"]
    result = {
        "n_items": len(index),
        "n_source_candidates": len(candidates),
        "n_sources": len(source_asin),
        "n_train_edges": len(train_pairs),
        "n_test_edges": len(test_pairs),
        "min_degree": args.min_degree,
        "test_ratio": 0.2,
        "seed": args.seed,
        "best_method": best,
        "random_recall@10": 10 / len(index),
        "methods": {
            row["method"]: {
                key: row[key]
                for key in [
                    "recall@10",
                    "recall@50",
                    "ndcg@10",
                    "map@10",
                    "coverage@10",
                    "cross_recall@10",
                    "same_recall@10",
                    "recall@10_ci95",
                ]
            }
            for row in rows
        },
        "notes": [
            "also_buy 是推荐位列表，不是购买篮；评估的是推荐位补全，不是购买预测。",
            "按源商品留出 20% 的 also_buy 目标，训练图不含这些边；also_view 图不切分，作为独立特征。",
            "Recall@10 的随机期望约为 10 / 商品数。",
        ],
        "outputs": {"metrics": str(OUT_METRICS), "figure": str(OUT_FIG)},
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    order = metrics_frame["method"].tolist()
    labels = [METHOD_LABELS[method] for method in order]
    x = np.arange(len(order))
    width = 0.38
    axes[0].bar(x - width / 2, metrics_frame["recall@10"], width, label="Recall@10", color="#4c72b0")
    axes[0].bar(x + width / 2, metrics_frame["recall@50"], width, label="Recall@50", color="#dd8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("Recall")
    axes[0].set_title("链接预测召回率")
    axes[0].legend()

    axes[1].bar(x - width / 2, metrics_frame["ndcg@10"], width, label="NDCG@10", color="#55a868")
    axes[1].bar(x + width / 2, metrics_frame["map@10"], width, label="MAP@10", color="#c44e52")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("指标")
    axes[1].set_title("排序质量")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(metrics_frame[["method_label", "recall@10", "recall@50", "ndcg@10", "map@10"]].to_string(index=False))
    print(f"最佳方法：{METHOD_LABELS[best]}；耗时 {time.perf_counter() - started:.1f} 秒")
    print(f"写入 {OUT_JSON}")


if __name__ == "__main__":
    main()
