"""聚类外部验证：训练集上学习簇，留出集上检验业务结果。

不是在同一批商品上做 KMeans 再算均值，而是：
1. 按 asin 70/30 切分商品；
2. 训练集拟合缩尾、标准化与 KMeans(k=4)；
3. 留出集按最近质心分配簇；
4. 在留出集上比较评分、评论量、verified、价格观测与 also_buy。

用法：
    python src/cluster_validate.py
"""

from __future__ import annotations

import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from .cluster import (
        CLUSTER_FEATURES,
        LOG_COLUMNS,
        add_review_aggregates,
        fill_price,
        load_products,
        setup_fonts,
    )
    from .config import FIGURES, SEED, TABLES
except ImportError:
    from cluster import (
        CLUSTER_FEATURES,
        LOG_COLUMNS,
        add_review_aggregates,
        fill_price,
        load_products,
        setup_fonts,
    )
    from config import FIGURES, SEED, TABLES

OUT_CSV = TABLES / "cluster_validation_outcomes.csv"
OUT_JSON = TABLES / "cluster_validation.json"
OUT_FIG = FIGURES / "fig45_cluster_validation.png"
K = 4
TEST_FRACTION = 0.3


def transform(train: pd.DataFrame, test: pd.DataFrame):
    matrix_train = train[CLUSTER_FEATURES].astype(float).fillna(0.0).copy()
    matrix_test = test[CLUSTER_FEATURES].astype(float).fillna(0.0).copy()
    for column in LOG_COLUMNS:
        matrix_train[column] = np.log1p(matrix_train[column].clip(lower=0))
        matrix_test[column] = np.log1p(matrix_test[column].clip(lower=0))
    low = matrix_train.quantile(0.01)
    high = matrix_train.quantile(0.99)
    matrix_train = matrix_train.clip(low, high, axis=1)
    matrix_test = matrix_test.clip(low, high, axis=1)
    keep = matrix_train.std(ddof=0) > 0
    matrix_train = matrix_train.loc[:, keep]
    matrix_test = matrix_test.loc[:, keep]
    scaler = StandardScaler()
    return (
        scaler.fit_transform(matrix_train.to_numpy(dtype=float)),
        scaler.transform(matrix_test.to_numpy(dtype=float)),
        list(matrix_train.columns),
    )


def main() -> None:
    setup_fonts()
    products, _ = load_products()
    products = add_review_aggregates(products)
    products = fill_price(products)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(products))
    n_test = int(round(len(products) * TEST_FRACTION))
    test_index = np.sort(order[:n_test])
    train_index = np.sort(order[n_test:])
    train = products.iloc[train_index].reset_index(drop=True)
    test = products.iloc[test_index].reset_index(drop=True)
    train_matrix, test_matrix, names = transform(train, test)

    model = KMeans(n_clusters=K, n_init=30, random_state=SEED)
    train_labels = model.fit_predict(train_matrix)
    distance = ((test_matrix[:, None, :] - model.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
    test_labels = distance.argmin(axis=1)
    test = test.copy()
    test["cluster"] = test_labels
    test["price_observed"] = test["price"].notna()
    train_silhouette = float(silhouette_score(train_matrix, train_labels, random_state=SEED))

    rows = []
    for cluster in range(K):
        part = test.loc[test["cluster"] == cluster]
        rows.append({
            "cluster": cluster,
            "n_products": int(len(part)),
            "mean_overall": round(float(part["mean_overall"].mean()), 4),
            "mean_n_reviews": round(float(part["n_reviews"].mean()), 1),
            "verified_share": round(float(part["verified_share"].mean()), 4),
            "price_observed": round(float(part["price"].notna().mean()), 4),
            "mean_also_buy": round(float(part["also_buy_len"].mean()), 1),
        })

    tests = {}
    for metric in ["mean_overall", "n_reviews", "verified_share", "price_observed", "also_buy_len"]:
        groups = [test.loc[test["cluster"] == cluster, metric].dropna().to_numpy() for cluster in range(K)]
        statistic, p_value = kruskal(*groups)
        n, k = len(test), K
        tests[metric] = {
            "statistic": round(float(statistic), 2),
            "p_value": float(p_value),
            "epsilon_squared": round(float((statistic - k + 1) / (n - k)), 4),
        }

    outcomes = pd.DataFrame(rows)
    outcomes.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar(outcomes["cluster"].astype(str), outcomes["mean_overall"], color="#4c72b0")
    axes[0].set_ylim(1, 5)
    axes[0].set_xlabel("簇")
    axes[0].set_ylabel("留出集平均评分")
    axes[0].set_title("留出集评分")
    axes[1].bar(outcomes["cluster"].astype(str), outcomes["mean_n_reviews"], color="#dd8452")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("簇")
    axes[1].set_ylabel("留出集平均评论数（对数）")
    axes[1].set_title("留出集评论量")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(figure)

    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED,
        "n_products": int(len(products)),
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "k": K,
        "train_silhouette": round(train_silhouette, 4),
        "outcomes": outcomes.to_dict("records"),
        "kruskal": tests,
        "method": "训练集拟合缩尾/标准化/KMeans，留出集按最近质心分配；结果只在留出集统计",
        "outputs": {"csv": str(OUT_CSV), "figure": str(OUT_FIG)},
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(outcomes.to_string(index=False))
    print(json.dumps(tests, ensure_ascii=False, indent=2))
    print(f"写入 {OUT_JSON}")


if __name__ == "__main__":
    main()
