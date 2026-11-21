"""P5 商品聚类：游戏本体商品的分群、k 选择与 bootstrap 稳定性。

一行一个商品，特征来自商品元数据与评论聚合，不使用评分或任何由标签派生的列。
评分均值只在聚类完成后做事后画像，不进入特征矩阵。
价格缺失 83.44%，若把缺失指示放进距离会让聚类退化成“有没有标价”两群，
因此价格不参与聚类特征，只在簇画像里事后统计。

用法：
    python src/cluster.py --force
    python src/cluster.py --k 4 --bootstrap 50 --force
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from .build_features import category_depth, category_leaf, clean_main_cat, clean_text
    from .config import (
        DATA_PROCESSED,
        FEATURES_PARQUET,
        FIGURES,
        FONT_CANDIDATES,
        SEED,
        TABLES,
    )
    from .parse import _hardware
except ImportError:
    from build_features import category_depth, category_leaf, clean_main_cat, clean_text
    from config import (
        DATA_PROCESSED,
        FEATURES_PARQUET,
        FIGURES,
        FONT_CANDIDATES,
        SEED,
        TABLES,
    )
    from parse import _hardware

META_PARQUET = DATA_PROCESSED / "meta.parquet"
SCOPE_PARQUET = DATA_PROCESSED / "scope.parquet"
SUMMARY_JSON = TABLES / "cluster_summary.json"
PROFILES_CSV = TABLES / "cluster_profiles.csv"
STABILITY_CSV = TABLES / "cluster_stability.csv"

K_RANGE = list(range(2, 11))
SILHOUETTE_SAMPLE = 5_000
PCA_SAMPLE = 8_000
BOOTSTRAP_FRACTION = 0.8
MIN_ARI = 0.6
MIN_CLUSTER_SHARE = 0.02

LOG_COLUMNS = [
    "n_reviews",
    "mean_review_len",
    "mean_vote",
    "also_buy_len",
    "also_view_len",
    "feature_count",
    "description_len",
    "rank_count",
    "image_count_meta",
    "category_leaf_freq",
    "main_cat_freq",
]

CLUSTER_FEATURES = [
    "n_reviews",
    "mean_review_len",
    "verified_share",
    "mean_vote",
    "also_buy_len",
    "also_view_len",
    "feature_count",
    "description_len",
    "category_depth",
    "rank_count",
    "image_count_meta",
    "tech1_present",
    "tech2_present",
    "category_leaf_freq",
    "main_cat_freq",
]

PROFILE_FEATURES = ["log_price", "price_missing"] + CLUSTER_FEATURES


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def short(text: str, limit: int = 60) -> str:
    text = "" if not isinstance(text, str) else text.strip()
    if not text:
        return "(无标题)"
    return text if len(text) <= limit else text[: limit - 1] + "…"


def load_products():
    meta = pq.read_table(META_PARQUET).to_pandas()
    scope = pq.read_table(SCOPE_PARQUET, columns=["asin", "in_scope"]).to_pandas()
    assert scope["asin"].is_unique, "scope 表 asin 不唯一"
    in_scope = set(scope.loc[scope["in_scope"], "asin"])
    base = meta.loc[meta["asin"].isin(in_scope)].copy()
    assert len(base) > 30_000, "范围内商品过少"

    products = pd.DataFrame(
        {
            "asin": base["asin"],
            "title": base["title"].map(clean_text),
            "main_cat": base["main_cat"].map(clean_main_cat).fillna("(缺失)"),
            "category_depth": base["category"].map(category_depth),
            "price": base["price"],
            "price_missing": base["price"].isna(),
            "log_price": np.log1p(base["price"]),
            "also_buy_len": base["also_buy"].map(len),
            "also_view_len": base["also_view"].map(len),
            "feature_count": base["feature"].map(len),
            "description_len": base["description"].str.len(),
            "rank_count": base["rank"].map(len),
            "image_count_meta": base["imageURL"].map(len),
            "tech1_present": base["tech1"].notna(),
            "tech2_present": base["tech2"].notna(),
            "category_leaf": base["category"].map(category_leaf),
        }
    )
    products["category_leaf_freq"] = products["category_leaf"].map(
        products["category_leaf"].value_counts()
    )
    products["category_leaf_freq"] = products["category_leaf_freq"].fillna(0.0)
    products["main_cat_freq"] = products["main_cat"].map(
        products["main_cat"].value_counts()
    )
    products["main_cat_freq"] = products["main_cat_freq"].fillna(0.0)
    return products, in_scope


def add_review_aggregates(products: pd.DataFrame) -> pd.DataFrame:
    table = pq.read_table(
        FEATURES_PARQUET,
        columns=["asin", "in_scope", "overall", "review_len", "verified", "vote_count"],
    ).to_pandas()
    table = table.loc[table["in_scope"]]
    grouped = (
        table.groupby("asin", observed=True)
        .agg(
            n_reviews=("overall", "size"),
            mean_review_len=("review_len", "mean"),
            verified_share=("verified", "mean"),
            mean_vote=("vote_count", "mean"),
            mean_overall=("overall", "mean"),
        )
        .reset_index()
    )
    products = products.merge(grouped, on="asin", how="left")
    products["n_reviews"] = products["n_reviews"].fillna(0)
    products["mean_review_len"] = products["mean_review_len"].fillna(0.0)
    products["verified_share"] = products["verified_share"].fillna(0.0)
    products["mean_vote"] = products["mean_vote"].fillna(0.0)
    return products


def fill_price(products: pd.DataFrame) -> pd.DataFrame:
    median = products.groupby("main_cat", observed=True)["log_price"].transform("median")
    products["log_price"] = products["log_price"].fillna(median)
    products["log_price"] = products["log_price"].fillna(products["log_price"].median())
    return products


def build_matrix(products: pd.DataFrame):
    profile_frame = products[PROFILE_FEATURES].astype(float).fillna(0.0).copy()
    matrix_frame = profile_frame[CLUSTER_FEATURES].copy()
    for column in LOG_COLUMNS:
        matrix_frame[column] = np.log1p(matrix_frame[column].clip(lower=0))
    # 1% 与 99% 分位缩尾，避免极端商品单独成簇
    matrix_frame = matrix_frame.clip(
        matrix_frame.quantile(0.01), matrix_frame.quantile(0.99), axis=1
    )
    # 缩尾后方差为 0 的列（例如极少出现的二值标记）不参与聚类
    matrix_frame = matrix_frame.loc[:, matrix_frame.std(ddof=0) > 0]
    assert not np.isnan(matrix_frame.to_numpy()).any(), "聚类特征存在缺失值"
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix_frame.to_numpy(dtype=float))
    assert abs(float(scaled.mean())) < 1e-6, "标准化后全局均值不为 0"
    assert abs(float(scaled.std()) - 1.0) < 1e-6, "标准化后全局标准差不为 1"
    return scaled, profile_frame, list(matrix_frame.columns)


def choose_k(matrix, k_override=None):
    rows = []
    sample = np.random.default_rng(SEED).choice(
        len(matrix), size=min(SILHOUETTE_SAMPLE, len(matrix)), replace=False
    )
    min_size = int(np.ceil(MIN_CLUSTER_SHARE * len(matrix)))
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        labels = model.fit_predict(matrix)
        sizes = np.bincount(labels, minlength=k)
        silhouette = silhouette_score(
            matrix[sample], labels[sample], random_state=SEED
        )
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette),
                "min_size": int(sizes.min()),
                "valid": bool(sizes.min() >= min_size),
            }
        )
        print(
            f"k={k}: inertia={model.inertia_:.1f}, silhouette={silhouette:.4f}, "
            f"min_size={sizes.min()}"
        )
    table = pd.DataFrame(rows)
    if k_override is not None:
        return table, k_override
    valid = table.loc[table["valid"]]
    assert len(valid) > 0, "没有满足最小簇规模的 k"
    chosen_k = int(valid.loc[valid["silhouette"].idxmax(), "k"])
    return table, chosen_k


def bootstrap_stability(matrix, base_labels, k, n_bootstrap):
    rng = np.random.default_rng(SEED)
    n = len(matrix)
    size = int(round(n * BOOTSTRAP_FRACTION))
    rows = []
    for index in range(1, n_bootstrap + 1):
        sample = rng.choice(n, size=size, replace=False)
        model = KMeans(n_clusters=k, n_init=30, random_state=SEED + index)
        labels = model.fit_predict(matrix[sample])
        ari = adjusted_rand_score(base_labels[sample], labels)
        rows.append(
            {
                "bootstrap": index,
                "ari": round(float(ari), 4),
                "k": k,
                "n_products": int(size),
            }
        )
        if index % 10 == 0:
            print(f"bootstrap {index}/{n_bootstrap}")
    return pd.DataFrame(rows)


def build_profiles(products, labels, profile_frame, feature_names, k):
    frame = products.copy()
    frame["cluster"] = labels
    cluster_means = profile_frame.groupby(labels).mean()

    rows = []
    names = []
    dominants = []
    reviews_rank = cluster_means["n_reviews"].rank(ascending=False, method="first")
    buy_rank = cluster_means["also_buy_len"].rank(ascending=False, method="first")
    top_cut = max(1, int(round(k * 0.25)))
    bottom_cut = max(1, int(round(k * 0.25)))
    for cluster in range(k):
        part = frame.loc[frame["cluster"] == cluster]
        representatives = part.sort_values("n_reviews", ascending=False)["title"]
        volume = (
            "高热度"
            if reviews_rank[cluster] <= top_cut
            else ("长尾" if reviews_rank[cluster] > k - bottom_cut else "中等热度")
        )
        buy_tag = (
            "共购密集"
            if buy_rank[cluster] <= top_cut
            else ("共购稀疏" if buy_rank[cluster] > k - bottom_cut else None)
        )
        digital_share = float(
            part["title"]
            .fillna("")
            .str.contains("download|digital code", case=False, regex=True)
            .mean()
        )
        tags = [volume] + ([buy_tag] if buy_tag else [])
        if digital_share > 0.5:
            tags.append("数字版")
        names.append("·".join(tags))
        dominants.append(str(part["main_cat"].value_counts().index[0]))
        rows.append(
            {
                "cluster": cluster,
                "size": int(len(part)),
                "share": round(len(part) / len(frame), 4),
                "mean_overall": round(float(part["mean_overall"].mean()), 4),
                "price_median_usd": (
                    round(float(part["price"].median()), 2)
                    if part["price"].notna().any()
                    else float("nan")
                ),
                "n_price_observed": int(part["price"].notna().sum()),
                "price_missing_share": round(float(part["price_missing"].mean()), 4),
                "top_titles": "; ".join(short(title, 60) for title in representatives.head(3)),
            }
        )
    profiles = pd.DataFrame(rows)

    seen: dict[str, int] = {}
    unique_names = []
    for index, name in enumerate(names):
        if name in seen:
            name = f"{name}·{short(dominants[index], 12)}"
        if name in seen:
            name = f"{name}-{index + 1}"
        seen[name] = seen.get(name, 0) + 1
        unique_names.append(name)
    profiles["name"] = unique_names

    for feature in feature_names:
        means = cluster_means[feature].reindex(profiles["cluster"]).to_numpy()
        profiles[f"mean_{feature}"] = np.round(means, 4)
    profiles["mean_log_price"] = np.round(
        cluster_means["log_price"].reindex(profiles["cluster"]).to_numpy(), 4
    )

    columns = [
        "cluster", "size", "share", "name", "mean_overall",
        "price_median_usd", "n_price_observed", "price_missing_share",
        "mean_log_price", "top_titles",
    ] + [f"mean_{feature}" for feature in feature_names]
    return profiles[columns]


def plot_k_selection(table, chosen_k, path_inertia, path_silhouette):
    plt.figure(figsize=(7, 4))
    plt.plot(table["k"], table["inertia"], marker="o")
    plt.axvline(chosen_k, color="red", linestyle="--", label=f"选定 k={chosen_k}")
    plt.xlabel("簇数 k")
    plt.ylabel("簇内平方和 inertia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_inertia, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(table["k"], table["silhouette"], marker="o", color="#55a868")
    plt.axvline(chosen_k, color="red", linestyle="--", label=f"选定 k={chosen_k}")
    plt.xlabel("簇数 k")
    plt.ylabel("轮廓系数 silhouette")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_silhouette, dpi=200)
    plt.close()


def plot_clusters(matrix, labels, profiles, path):
    sample = np.random.default_rng(SEED).choice(
        len(matrix), size=min(PCA_SAMPLE, len(matrix)), replace=False
    )
    coordinates = PCA(n_components=2, random_state=SEED).fit_transform(matrix[sample])
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        coordinates[:, 0], coordinates[:, 1], c=labels[sample],
        cmap="tab10", s=6, alpha=0.6,
    )
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=scatter.cmap(scatter.norm(i)))
        for i in range(len(profiles))
    ]
    plt.legend(handles, list(profiles["name"]), title="簇", fontsize=8)
    plt.xlabel("主成分 1")
    plt.ylabel("主成分 2")
    plt.title("商品聚类的 PCA 二维投影")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_profiles(profiles, path):
    features = [
        "n_reviews", "log_price", "mean_review_len", "verified_share",
        "also_buy_len", "also_view_len", "feature_count", "description_len",
        "category_depth", "rank_count", "image_count_meta", "price_missing",
    ]
    features = [name for name in features if f"mean_{name}" in profiles.columns]
    # 转置成“特征 × 簇”，与坐标轴标签一致
    values = profiles[[f"mean_{name}" for name in features]].to_numpy(dtype=float).T
    centered = values - values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    z_scores = centered / scale
    plt.figure(figsize=(max(7, 1.4 * len(profiles) + 3), 6))
    image = plt.imshow(z_scores, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    plt.colorbar(image, label="簇均值相对全体均值的标准化偏差")
    plt.xticks(range(len(profiles)), profiles["name"], rotation=30, ha="right")
    plt.yticks(range(len(features)), features)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            plt.text(
                col, row, f"{values[row, col]:.2f}", ha="center", va="center", fontsize=7
            )
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_sizes(profiles, path):
    plt.figure(figsize=(8, 4))
    plt.bar(profiles["name"], profiles["size"], color="#4c72b0")
    for index, value in enumerate(profiles["size"]):
        plt.text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("商品数")
    plt.title("各簇商品数量")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_bootstrap(stability, path):
    plt.figure(figsize=(7, 4))
    plt.hist(stability["ari"], bins=15, color="#c44e52")
    plt.axvline(MIN_ARI, color="black", linestyle="--", label=f"门槛 ARI={MIN_ARI}")
    plt.xlabel("调整兰德指数 ARI")
    plt.ylabel("bootstrap 次数")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_price_reviews(products, labels, profiles, path):
    sample = np.random.default_rng(SEED).choice(
        len(products), size=min(PCA_SAMPLE, len(products)), replace=False
    )
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        np.expm1(products["log_price"].to_numpy()[sample]),
        products["n_reviews"].to_numpy()[sample] + 1,
        c=labels[sample], cmap="tab10", s=8, alpha=0.6,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("价格（美元，对数刻度）")
    plt.ylabel("评论数 + 1（对数刻度）")
    plt.title("价格与评论热度的事后画像（颜色为簇）")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=scatter.cmap(scatter.norm(i)))
        for i in range(len(profiles))
    ]
    plt.legend(handles, list(profiles["name"]), title="簇", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="游戏本体商品聚类")
    parser.add_argument("--k", type=int, default=None, help="手动指定簇数，默认按轮廓系数选")
    parser.add_argument("--bootstrap", type=int, default=50, help="bootstrap 次数")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if SUMMARY_JSON.exists() and not args.force:
        raise SystemExit(f"输出已存在，加 --force 覆盖: {SUMMARY_JSON}")
    assert args.bootstrap >= 50, "bootstrap 至少 50 次"
    if args.k is not None:
        assert 2 <= args.k <= 10, "k 必须在 2 到 10 之间"

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    started = time.perf_counter()

    products, in_scope = load_products()
    products = add_review_aggregates(products)
    products = fill_price(products)
    print(f"商品 {len(products):,} 件，评论聚合完成")

    matrix, profile_frame, feature_names = build_matrix(products)
    print(f"聚类特征 {len(feature_names)} 个，样本 {len(matrix):,}")
    table, chosen_k = choose_k(matrix, args.k)
    print(f"选定 k={chosen_k}")

    model = KMeans(n_clusters=chosen_k, n_init=20, random_state=SEED)
    labels = model.fit_predict(matrix)
    counts = np.bincount(labels, minlength=chosen_k)
    assert len(counts) == chosen_k and counts.min() > 0, "存在空簇"
    assignments = products[["asin"]].copy()
    assignments["cluster"] = labels.astype(np.int8)
    assignments.to_parquet(TABLES / "cluster_assignments.parquet", index=False)

    stability = bootstrap_stability(matrix, labels, chosen_k, args.bootstrap)
    ari_mean = float(stability["ari"].mean())
    ari_ci = np.percentile(stability["ari"], [2.5, 97.5]).tolist()

    profiles = build_profiles(products, labels, profile_frame, feature_names, chosen_k)
    profiles.to_csv(PROFILES_CSV, index=False, encoding="utf-8-sig")
    stability.to_csv(STABILITY_CSV, index=False, encoding="utf-8-sig")

    plot_k_selection(
        table, chosen_k,
        FIGURES / "fig20_elbow.png", FIGURES / "fig21_silhouette.png",
    )
    plot_clusters(matrix, labels, profiles, FIGURES / "fig22_clusters_pca.png")
    plot_profiles(profiles, FIGURES / "fig23_profile_heatmap.png")
    plot_sizes(profiles, FIGURES / "fig24_cluster_sizes.png")
    plot_bootstrap(stability, FIGURES / "fig25_bootstrap_ari.png")
    plot_price_reviews(products, labels, profiles, FIGURES / "fig26_price_reviews.png")

    scope_only = set(products["asin"]) <= in_scope
    standardized = bool(
        abs(float(matrix.mean())) < 1e-6 and abs(float(matrix.std()) - 1.0) < 1e-6
    )
    assertions = {
        "k_ge_2": bool(chosen_k >= 2),
        "each_cluster_nonempty": bool(counts.min() > 0),
        "ari_ge_0_6": bool(ari_mean >= MIN_ARI),
        "scope_only": bool(scope_only),
        "features_standardized": standardized,
        "price_profile_complete": bool(
            profiles[
                ["price_median_usd", "n_price_observed", "price_missing_share"]
            ].notna().all().all()
        ),
    }
    assert scope_only, "聚类样本中出现范围外商品"
    assert standardized, "特征未标准化"
    assert assertions["price_profile_complete"], "簇画像价格列存在缺失"

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "seed": SEED,
        "input": {
            "n_products": int(len(products)),
            "feature_names": feature_names,
            "source": str(FEATURES_PARQUET),
        },
        "k_selection": {
            "k": table["k"].tolist(),
            "inertia": table["inertia"].round(2).tolist(),
            "silhouette": table["silhouette"].round(4).tolist(),
            "min_size": table["min_size"].tolist(),
            "chosen_k": int(chosen_k),
            "method": "轮廓系数最大，且每个簇不少于样本的 2%",
        },
        "model": "KMeans(n_init=20, random_state=42)",
        "stability": {
            "n_bootstrap": args.bootstrap,
            "ari_mean": round(ari_mean, 4),
            "ari_ci95": [round(float(value), 4) for value in ari_ci],
            "min_required": MIN_ARI,
            "bootstrap_fraction": BOOTSTRAP_FRACTION,
        },
        "cluster_sizes": [int(value) for value in counts],
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "assertions": assertions,
        "hardware": _hardware(),
        "outputs": {
            "profiles": str(PROFILES_CSV),
            "stability": str(STABILITY_CSV),
        },
    }
    SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    assert assertions["ari_ge_0_6"], f"ARI 均值 {ari_mean:.4f} 低于 0.6"
    print(f"完成，k={chosen_k}，ARI {ari_mean:.4f}，用时 {summary['runtime_seconds']} 秒")


if __name__ == "__main__":
    main()
