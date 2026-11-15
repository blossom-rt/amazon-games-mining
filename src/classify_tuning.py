"""分类超参数搜索：嵌套 5 折 + 内层 3 折 RandomizedSearch。

外层仍按 asin 分组做 5 折；每折只用训练部分做 3 折随机搜索，选出的参数
再在外层验证折评估。文本哈希与 TF-IDF 在外层训练折拟合，内层搜索复用
这份表示，因此内层验证折只影响参数选择，不进入外层测试。
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import loguniform
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold

try:
    from .config import FEATURES_PARQUET, SEED, TABLES
    from .parse import _hardware
except ImportError:
    from config import FEATURES_PARQUET, SEED, TABLES
    from parse import _hardware

OUT_JSON = TABLES / "classify_tuning.json"
OUT_CSV = TABLES / "classify_tuning_metrics.csv"
N_HASH = 2**18
OUTER_SPLITS = 5
INNER_SPLITS = 3
N_ITER = 6
SAMPLE_SIZE = 80_000


def main() -> None:
    started = time.perf_counter()
    rng = np.random.default_rng(SEED)
    columns = ["asin", "reviewText", "summary", "label", "in_scope"]
    frame = pq.read_table(FEATURES_PARQUET, columns=columns).to_pandas()
    frame = frame.loc[frame["in_scope"]].reset_index(drop=True)
    if len(frame) > SAMPLE_SIZE:
        positions = rng.choice(len(frame), size=SAMPLE_SIZE, replace=False)
        frame = frame.iloc[np.sort(positions)].reset_index(drop=True)

    y = frame["label"].to_numpy(dtype=np.int8)
    groups = frame["asin"].astype(str).to_numpy()
    text = (
        frame["reviewText"].fillna("").astype(str)
        + " "
        + frame["summary"].fillna("").astype(str)
    ).str.slice(0, 3000).tolist()

    counts = HashingVectorizer(
        n_features=N_HASH,
        alternate_sign=False,
        norm=None,
        token_pattern=r"(?u)\b\w\w+\b",
    ).transform(text)
    print(f"调参样本 {len(y):,}，正类 {y.mean():.4f}，哈希矩阵 {counts.shape}")

    param_distributions = {
        "alpha": loguniform(1e-6, 1e-4),
        "average": [True, False],
        "max_iter": [10, 15, 20],
        "tol": [None, 1e-3],
    }
    outer = StratifiedGroupKFold(
        n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED
    )
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(
        outer.split(np.zeros(len(y)), y, groups), start=1
    ):
        fold_started = time.perf_counter()
        tfidf = TfidfTransformer(sublinear_tf=True)
        train_x = tfidf.fit_transform(counts[train_idx])
        valid_x = tfidf.transform(counts[valid_idx])

        default = SGDClassifier(
            loss="hinge",
            alpha=1e-5,
            max_iter=15,
            tol=1e-3,
            class_weight="balanced",
            random_state=SEED + fold,
        )
        default.fit(train_x, y[train_idx])
        default_pred = default.predict(valid_x)

        search = RandomizedSearchCV(
            SGDClassifier(
                loss="hinge",
                class_weight="balanced",
                random_state=SEED + fold,
            ),
            param_distributions=param_distributions,
            n_iter=N_ITER,
            scoring="f1_macro",
            cv=StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + fold),
            random_state=SEED + fold,
            n_jobs=1,
            refit=True,
        )
        search.fit(train_x, y[train_idx])
        tuned = search.best_estimator_
        tuned_pred = tuned.predict(valid_x)
        rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "inner_best_f1": float(search.best_score_),
                "default_f1": float(f1_score(y[valid_idx], default_pred, average="macro")),
                "tuned_f1": float(f1_score(y[valid_idx], tuned_pred, average="macro")),
                "tuned_precision": float(precision_score(y[valid_idx], tuned_pred, average="macro", zero_division=0)),
                "tuned_recall": float(recall_score(y[valid_idx], tuned_pred, average="macro", zero_division=0)),
                "best_params": json.dumps(
                    {key: (value if not isinstance(value, np.floating) else float(value))
                     for key, value in search.best_params_.items()},
                    ensure_ascii=False,
                ),
                "seconds": round(time.perf_counter() - fold_started, 2),
            }
        )
        print(
            f"折 {fold}: 默认 {rows[-1]['default_f1']:.4f} -> 调参 {rows[-1]['tuned_f1']:.4f}，"
            f"参数 {search.best_params_}"
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    assert len(metrics) == OUTER_SPLITS, "外层折数不足"
    assert metrics["tuned_f1"].between(0, 1).all(), "F1 越界"
    assert (metrics["tuned_f1"] >= 0).all(), "F1 为负"
    delta = float(metrics["tuned_f1"].mean() - metrics["default_f1"].mean())
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED,
        "sample_size": int(len(y)),
        "positive_rate": float(y.mean()),
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "n_iter": N_ITER,
        "default": {
            "macro_f1_mean": float(metrics["default_f1"].mean()),
            "macro_f1_std": float(metrics["default_f1"].std(ddof=1)),
        },
        "tuned": {
            "macro_f1_mean": float(metrics["tuned_f1"].mean()),
            "macro_f1_std": float(metrics["tuned_f1"].std(ddof=1)),
            "precision_mean": float(metrics["tuned_precision"].mean()),
            "recall_mean": float(metrics["tuned_recall"].mean()),
        },
        "delta_mean": delta,
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "hardware": _hardware(),
        "notes": [
            "外层 5 折按 asin 分组，内层 3 折只在训练折调参；",
            "TF-IDF 在外层训练折拟合，内层验证折不参与外层测试；",
            "调参只针对线性 SVM（SGD hinge），默认参数作为对照。",
        ],
        "outputs": {"metrics": str(OUT_CSV)},
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"默认 F1 {summary['default']['macro_f1_mean']:.4f}，"
        f"调参 F1 {summary['tuned']['macro_f1_mean']:.4f}，"
        f"提升 {delta:+.4f}，耗时 {summary['runtime_seconds']} 秒"
    )


if __name__ == "__main__":
    main()
