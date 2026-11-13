"""P5 聚类增强：KMeans / GMM / HDBSCAN / 有价子集对比。

主聚类仍以 KMeans(k=4) 为基线；本脚本只做补充实验，不覆盖 cluster.py 的
主产物。重点回答两个问题：
1. 换成 GMM 或 HDBSCAN 后，簇结构是否更合理（轮廓、噪声比例、簇规模）；
2. 只在“有观测价格”的商品上聚类，价格是否呈现更清楚的梯度。

用法：
    python src/cluster_compare.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

try:
    from .cluster import (
        CLUSTER_FEATURES, add_review_aggregates, build_matrix, load_products,
    )
    from .config import FIGURES, FONT_CANDIDATES, SEED, TABLES
except ImportError:
    from cluster import (
        CLUSTER_FEATURES, add_review_aggregates, build_matrix, load_products,
    )
    from config import FIGURES, FONT_CANDIDATES, SEED, TABLES

OUT_JSON = TABLES / "cluster_compare.json"
OUT_PROFILES = TABLES / "cluster_compare_profiles.csv"
OUT_FIG = FIGURES / "fig39_cluster_methods.png"
PRICE_K = 4


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False


def profile_table(products: pd.DataFrame, labels: np.ndarray, method: str) -> pd.DataFrame:
    frame = products.copy()
    frame["cluster"] = labels
    frame = frame.loc[frame["cluster"] >= 0]
    rows = []
    for cluster, part in frame.groupby("cluster", observed=True):
        prices = part["price"].dropna()
        rows.append(
            {
                "method": method,
                "cluster": int(cluster),
                "size": int(len(part)),
                "price_observed": int(len(prices)),
                "price_median_usd": round(float(prices.median()), 2) if len(prices) else None,
                "mean_overall": round(float(part["mean_overall"].mean()), 3)
                if "mean_overall" in part else None,
                "mean_n_reviews": round(float(part["n_reviews"].mean()), 1),
                "mean_also_buy_len": round(float(part["also_buy_len"].mean()), 2),
                "top_main_cat": part["main_cat"].value_counts().index[0]
                if len(part) else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "cluster"])


def evaluate(
    matrix: np.ndarray, products: pd.DataFrame, method: str
) -> tuple[dict, pd.DataFrame | None, np.ndarray]:
    started = time.perf_counter()
    if method == "kmeans":
        model = KMeans(n_clusters=4, n_init=20, random_state=SEED)
        labels = model.fit_predict(matrix)
        extra = {"inertia": round(float(model.inertia_), 1)}
    elif method == "gmm":
        model = GaussianMixture(
            n_components=4, covariance_type="diag", n_init=3,
            random_state=SEED, reg_covar=1e-3,
        )
        labels = model.fit_predict(matrix)
        extra = {"bic": round(float(model.bic(matrix)), 1)}
    elif method == "hdbscan":
        model = HDBSCAN(
            min_cluster_size=800, min_samples=40, metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = model.fit_predict(matrix)
        extra = {
            "noise_share": round(float((labels == -1).mean()), 4),
            "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
        }
    else:
        raise ValueError(method)

    mask = labels >= 0
    n_clusters = len(set(labels[mask]))
    if mask.sum() > 2 and n_clusters > 1:
        sample_size = min(20_000, int(mask.sum()))
        rng = np.random.default_rng(SEED)
        index = rng.choice(np.flatnonzero(mask), size=sample_size, replace=False)
        silhouette = float(silhouette_score(matrix[index], labels[index]))
    else:
        silhouette = None
    extra.update(
        {
            "method": method,
            "n": int(len(labels)),
            "n_clusters": int(n_clusters),
            "silhouette": round(silhouette, 4) if silhouette is not None else None,
            "seconds": round(time.perf_counter() - started, 1),
        }
    )
    profiles = profile_table(products, labels, method)
    return extra, profiles, labels


def plot_comparison(table: pd.DataFrame, profiles: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar(table["method"], table["silhouette"], color="#4c72b0")
    axes[0].set_ylabel("轮廓系数")
    axes[0].set_title("不同聚类方法的轮廓系数")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.25)

    for method, part in profiles.groupby("method"):
        if method == "hdbscan":
            continue
        axes[1].plot(
            np.arange(len(part)), part["price_median_usd"],
            marker="o", label=method,
        )
    axes[1].set_xlabel("簇编号")
    axes[1].set_ylabel("价格中位数（美元，有观测价格）")
    axes[1].set_title("簇价格梯度（不含 HDBSCAN 噪声）")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="聚类方法补充对比")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if OUT_JSON.exists() and not args.force:
        raise SystemExit(f"输出已存在，加 --force 覆盖: {OUT_JSON}")

    setup_fonts()
    started = time.perf_counter()
    products, _ = load_products()
    products = add_review_aggregates(products)
    matrix, profile_frame, feature_names = build_matrix(products)
    print(f"商品 {len(products):,}，特征 {len(feature_names)}")

    records = []
    profiles = []
    for method in ["kmeans", "gmm", "hdbscan"]:
        extra, table, labels = evaluate(matrix, products, method)
        records.append(extra)
        if table is not None:
            profiles.append(table)
        print(f"{method}: {extra}")

    price_mask = products["price"].notna().to_numpy()
    price_products = products.loc[price_mask].reset_index(drop=True)
    price_matrix, _, price_features = build_matrix(price_products)
    price_model = KMeans(n_clusters=PRICE_K, n_init=20, random_state=SEED)
    price_labels = price_model.fit_predict(price_matrix)
    price_silhouette = float(
        silhouette_score(
            price_matrix,
            price_labels,
            sample_size=min(20_000, len(price_matrix)),
            random_state=SEED,
        )
    )
    price_profiles = profile_table(price_products, price_labels, "price_subset_kmeans")
    profiles.append(price_profiles)
    records.append(
        {
            "method": "price_subset_kmeans",
            "n": int(price_mask.sum()),
            "n_clusters": PRICE_K,
            "silhouette": round(price_silhouette, 4),
            "features": len(price_features),
            "seconds": None,
        }
    )
    print(f"有价子集 n={price_mask.sum():,}, silhouette={price_silhouette:.4f}")

    table = pd.DataFrame(records)
    profile_frame = pd.concat(profiles, ignore_index=True)
    OUT_PROFILES.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "cluster_compare_metrics.csv", index=False, encoding="utf-8-sig")
    profile_frame.to_csv(OUT_PROFILES, index=False, encoding="utf-8-sig")
    plot_comparison(table, profile_frame, OUT_FIG)

    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED,
        "n_products": int(len(products)),
        "n_price_observed": int(price_mask.sum()),
        "methods": table.to_dict(orient="records"),
        "notes": [
            "KMeans/GMM/HDBSCAN 使用同一 13 维标准化特征，价格缺失由品类中位数填补；",
            "有价子集只用 price 非空的商品，不再填补价格；",
            "HDBSCAN 的轮廓系数只在非噪声点上计算，噪声比例单独报告；",
            "本脚本不覆盖 cluster.py 的主产物，只用于方法对比。",
        ],
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "hardware": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "outputs": {
            "metrics": str(TABLES / "cluster_compare_metrics.csv"),
            "profiles": str(OUT_PROFILES),
            "figure": str(OUT_FIG),
        },
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"写入 {OUT_JSON}")


if __name__ == "__main__":
    main()
