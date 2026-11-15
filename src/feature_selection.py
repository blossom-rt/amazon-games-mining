"""特征选择与降维：方差、相关性、嵌入法、PCA 与 t-SNE。

在游戏本体宽表的数值/布尔特征上做三组筛选：
1) 方差为 0 的常量列；
2) |相关系数| > 0.95 的冗余列；
3) ExtraTrees 嵌入重要性前 15。
再比较全特征与筛选后特征的 5 折 HistGradientBoosting 宏 F1，并给出
PCA 解释方差与 t-SNE 二维投影（5,000 条抽样）。
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from .config import FEATURES_PARQUET, FIGURES, FONT_CANDIDATES, SEED, TABLES
    from .build_features import MODEL_FEATURES
except ImportError:
    from config import FEATURES_PARQUET, FIGURES, FONT_CANDIDATES, SEED, TABLES
    from build_features import MODEL_FEATURES

OUT_CSV = TABLES / "feature_selection.csv"
OUT_JSON = TABLES / "feature_selection.json"
OUT_FIG = FIGURES / "fig50_feature_selection.png"
SAMPLE = 50_000
TSNE_SAMPLE = 5_000


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column for column in MODEL_FEATURES
        if column in frame.columns
        and (pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]))
    ]


def cross_val_f1(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    scores = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for train_idx, valid_idx in splitter.split(x, y):
        model = HistGradientBoostingClassifier(
            max_iter=80, learning_rate=0.08, max_leaf_nodes=31,
            early_stopping=True, random_state=SEED,
        )
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[valid_idx])
        scores.append(f1_score(y[valid_idx], pred, average="macro"))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def main() -> None:
    plt.rcParams["font.sans-serif"] = FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False
    started = time.perf_counter()
    frame = pq.read_table(FEATURES_PARQUET).to_pandas()
    frame = frame.loc[frame["in_scope"]].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    if len(frame) > SAMPLE:
        positions = rng.choice(len(frame), size=SAMPLE, replace=False)
        frame = frame.iloc[np.sort(positions)].reset_index(drop=True)
    columns = numeric_columns(frame)
    x = frame[columns].astype(float)
    y = frame["label"].to_numpy(dtype=np.int8)
    print(f"样本 {len(frame):,}，候选数值/布尔特征 {len(columns)}")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    matrix = scaler.fit_transform(imputer.fit_transform(x))

    variances = matrix.var(axis=0)
    keep_variance = variances > 0
    variance_columns = [column for column, keep in zip(columns, keep_variance) if keep]
    matrix_variance = matrix[:, keep_variance]

    corr = np.corrcoef(matrix_variance, rowvar=False)
    drop_corr = set()
    for i in range(len(variance_columns)):
        for j in range(i + 1, len(variance_columns)):
            if abs(corr[i, j]) > 0.95:
                drop_corr.add(j)
    corr_columns = [
        column for index, column in enumerate(variance_columns) if index not in drop_corr
    ]
    corr_indices = [variance_columns.index(column) for column in corr_columns]
    matrix_corr = matrix_variance[:, corr_indices]

    forest = ExtraTreesClassifier(
        n_estimators=120, max_depth=12, min_samples_leaf=20,
        random_state=SEED, n_jobs=-1,
    )
    forest.fit(matrix_corr, y)
    order = np.argsort(forest.feature_importances_)[::-1]
    top_idx = order[:15]
    top_columns = [corr_columns[index] for index in top_idx]
    matrix_top = matrix_corr[:, top_idx]
    print(f"方差保留 {len(variance_columns)}，相关去冗余 {len(corr_columns)}，嵌入 top15")

    f1_all, std_all = cross_val_f1(matrix_corr, y)
    f1_top, std_top = cross_val_f1(matrix_top, y)

    pca = PCA(n_components=10, random_state=SEED)
    pca.fit(matrix_corr)
    explained = pca.explained_variance_ratio_

    tsne_idx = rng.choice(len(matrix_corr), size=min(TSNE_SAMPLE, len(matrix_corr)), replace=False)
    tsne = TSNE(
        n_components=2, perplexity=30, learning_rate="auto",
        init="pca", random_state=SEED,
    )
    embedding = tsne.fit_transform(matrix_corr[tsne_idx])

    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sample_size": int(len(frame)),
        "candidate_features": columns,
        "variance_kept": variance_columns,
        "correlation_kept": corr_columns,
        "top15_embedded": top_columns,
        "embedded_importance": {
            column: float(forest.feature_importances_[index])
            for column, index in zip(corr_columns, range(len(corr_columns)))
        },
        "model_comparison": {
            "all_after_corr": {"macro_f1_mean": f1_all, "macro_f1_std": std_all},
            "top15_embedded": {"macro_f1_mean": f1_top, "macro_f1_std": std_top},
        },
        "pca_explained_variance": [float(value) for value in explained],
        "pca_cumulative": [float(value) for value in np.cumsum(explained)],
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "notes": [
            "50,000 条分层抽样；只在训练折拟合的模型评估为对照，不是主分类结果；",
            "相关去冗余在抽样上计算，阈值 |r|>0.95；",
            "t-SNE 仅用于可视化，不进入模型。",
        ],
        "outputs": {"csv": str(OUT_CSV), "figure": str(OUT_FIG)},
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "feature": corr_columns,
            "importance": [float(forest.feature_importances_[i]) for i in range(len(corr_columns))],
            "selected_top15": [column in top_columns for column in corr_columns],
        }
    ).sort_values("importance", ascending=False).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].bar(range(1, len(explained) + 1), explained * 100, color="#4c72b0")
    axes[0].plot(range(1, len(explained) + 1), np.cumsum(explained) * 100, "o-", color="#c44e52")
    axes[0].set_xlabel("主成分")
    axes[0].set_ylabel("解释方差（%）")
    axes[0].set_title("PCA 前 10 个主成分")
    axes[0].grid(alpha=0.25)
    colors = np.where(y[tsne_idx] == 1, "#55a868", "#c44e52")
    axes[1].scatter(embedding[:, 0], embedding[:, 1], c=colors, s=5, alpha=0.6)
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].set_title("t-SNE 二维投影（5,000 条）")
    axes[1].grid(alpha=0.25)
    figure.suptitle("特征选择与降维（游戏本体数值特征）")
    figure.tight_layout()
    figure.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"全特征 F1 {f1_all:.4f}，top15 F1 {f1_top:.4f}")
    print(f"写入 {OUT_JSON}；耗时 {result['runtime_seconds']} 秒")


if __name__ == "__main__":
    main()
