"""类别不平衡处理对照：无处理、class_weight、随机欠采样。

20 万条分层抽样、按 asin 5 折；比较逻辑回归与线性 SVM 在原始训练分布、
类别权重、以及多数类随机欠采样下的宏 F1/PR-AUC/ROC-AUC。
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

try:
    from .config import FEATURES_PARQUET, TABLES, SEED
except ImportError:
    from config import FEATURES_PARQUET, TABLES, SEED

OUT_CSV = TABLES / "classify_imbalance.csv"
OUT_JSON = TABLES / "classify_imbalance.json"
N_HASH = 2**18
N_SPLITS = 5
SAMPLE_SIZE = 200_000


def main() -> None:
    rng = np.random.default_rng(SEED)
    frame = pq.read_table(
        FEATURES_PARQUET,
        columns=["asin", "reviewText", "summary", "label", "in_scope"],
    ).to_pandas()
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
    print(f"样本 {len(y):,}，正类 {y.mean():.4f}")

    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(np.zeros(len(y)), y, groups), start=1
    ):
        tfidf = TfidfTransformer(sublinear_tf=True)
        train_x = tfidf.fit_transform(counts[train_idx])
        valid_x = tfidf.transform(counts[valid_idx])
        y_train, y_valid = y[train_idx], y[valid_idx]

        # 随机欠采样：保留全部少数类，多数类随机抽到与少数类相同数量
        positive = np.flatnonzero(y_train == 1)
        negative = np.flatnonzero(y_train == 0)
        n_keep = min(len(positive), len(negative))
        undersample_idx = np.concatenate(
            [
                rng.choice(positive, n_keep, replace=False),
                rng.choice(negative, n_keep, replace=False),
            ]
        )
        rng.shuffle(undersample_idx)
        under_x = train_x[undersample_idx]
        under_y = y_train[undersample_idx]

        settings = [
            ("logistic", None, "原始分布", train_x, y_train),
            ("logistic", "balanced", "class_weight", train_x, y_train),
            ("logistic", None, "随机欠采样", under_x, under_y),
            ("svm", None, "原始分布", train_x, y_train),
            ("svm", "balanced", "class_weight", train_x, y_train),
            ("svm", None, "随机欠采样", under_x, under_y),
        ]
        for model_name, weight, strategy, x_train, y_fit in settings:
            model = SGDClassifier(
                loss="log_loss" if model_name == "logistic" else "hinge",
                alpha=1e-5,
                max_iter=8 if model_name == "logistic" else 15,
                tol=None if model_name == "logistic" else 1e-3,
                class_weight=weight,
                random_state=SEED + fold,
            )
            model.fit(x_train, y_fit)
            score = (
                model.predict_proba(valid_x)[:, 1]
                if model_name == "logistic"
                else model.decision_function(valid_x)
            )
            pred = (score >= 0.5).astype(int) if model_name == "logistic" else model.predict(valid_x)
            rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "strategy": strategy,
                    "train_positive": int((y_fit == 1).sum()),
                    "train_negative": int((y_fit == 0).sum()),
                    "train_positive_rate": float(y_fit.mean()),
                    "predicted_positive_rate": float(pred.mean()),
                    "macro_f1": float(f1_score(y_valid, pred, average="macro")),
                    "pr_auc": float(average_precision_score(y_valid, score)),
                    "roc_auc": float(roc_auc_score(y_valid, score)),
                }
            )
        print(f"折 {fold} 完成")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED,
        "sample_size": int(len(y)),
        "positive_rate": float(y.mean()),
        "n_folds": N_SPLITS,
        "aggregate": (
            metrics.groupby(["model", "strategy"])
            .agg(
                macro_f1_mean=("macro_f1", "mean"),
                macro_f1_std=("macro_f1", "std"),
                pr_auc_mean=("pr_auc", "mean"),
                roc_auc_mean=("roc_auc", "mean"),
                predicted_positive_rate=("predicted_positive_rate", "mean"),
                train_positive=("train_positive", "mean"),
                train_negative=("train_negative", "mean"),
            )
            .reset_index()
            .to_dict(orient="records")
        ),
        "notes": [
            "原始分布、class_weight 与随机欠采样在同一外层 5 折上比较；",
            "欠采样只作用于训练折，验证折保持原始分布；",
            "线性 SVM 的分数是 decision_function，阈值用于预测正类。",
        ],
        "outputs": {"csv": str(OUT_CSV)},
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.groupby(["model", "strategy"])["macro_f1"].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
