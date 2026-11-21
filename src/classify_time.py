"""时间维度回测：用早期评论训练，检查模型在后续年份是否漂移。

不是随机折，而是按评论年份切分：
- 训练：review_year <= 2014
- 测试：2015–2018，逐年报告
- 反向再训一次 2015+、测 2014 及以前，作为方向性对照

用法：
    python src/classify_time.py
    python src/classify_time.py --limit 300000
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from .classify import MAX_TEXT_CHARS, N_HASH
    from .config import FEATURES_PARQUET, FIGURES, FONT_CANDIDATES, SEED, TABLES
except ImportError:
    from classify import MAX_TEXT_CHARS, N_HASH
    from config import FEATURES_PARQUET, FIGURES, FONT_CANDIDATES, SEED, TABLES

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

OUT_CSV = TABLES / "classify_time_metrics.csv"
OUT_JSON = TABLES / "classify_time.json"
OUT_FIG = FIGURES / "fig43_time_drift.png"


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False


def load(limit: int) -> pd.DataFrame:
    frame = pq.read_table(
        FEATURES_PARQUET,
        columns=["asin", "label", "in_scope", "reviewText", "summary", "review_year"],
    ).to_pandas()
    frame = frame.loc[frame["in_scope"]].reset_index(drop=True)
    if limit < len(frame):
        frame = frame.sample(n=limit, random_state=SEED).reset_index(drop=True)
    assert frame["review_year"].between(1997, 2018).all(), "年份超出数据范围"
    return frame


def matrix(text: np.ndarray):
    vectorizer = HashingVectorizer(
        n_features=N_HASH,
        alternate_sign=False,
        norm=None,
        dtype=np.float32,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    return vectorizer.transform(text)


def fit_score(train, test, counts_train, counts_test):
    tfidf = TfidfTransformer(sublinear_tf=True)
    x_train = tfidf.fit_transform(counts_train)
    x_test = tfidf.transform(counts_test)
    model = SGDClassifier(
        loss="hinge", alpha=1e-5, max_iter=15, tol=1e-3,
        class_weight="balanced", random_state=SEED,
    )
    model.fit(x_train, train["label"].to_numpy())
    score = model.decision_function(x_test)
    pred = (score >= 0).astype(np.int8)
    return pred, score, model


def metrics(frame: pd.DataFrame, pred: np.ndarray, score: np.ndarray) -> dict:
    y = frame["label"].to_numpy()
    return {
        "n": int(len(frame)),
        "positive_rate": round(float(y.mean()), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "pr_auc": round(float(average_precision_score(y, score)), 4),
        "roc_auc": round(float(roc_auc_score(y, score)), 4),
    }


def run_direction(frame: pd.DataFrame, train_mask, label: str) -> tuple[list[dict], dict]:
    train, test = frame.loc[train_mask].reset_index(drop=True), frame.loc[~train_mask].reset_index(drop=True)
    assert len(train) > 10_000 and len(test) > 10_000, "时间切分后样本过少"
    text_train = (train["reviewText"].fillna("") + " " + train["summary"].fillna("")) \
        .str.slice(0, MAX_TEXT_CHARS).to_numpy()
    text_test = (test["reviewText"].fillna("") + " " + test["summary"].fillna("")) \
        .str.slice(0, MAX_TEXT_CHARS).to_numpy()
    started = time.perf_counter()
    counts_train, counts_test = matrix(text_train), matrix(text_test)
    pred, score, _ = fit_score(train, test, counts_train, counts_test)
    seconds = round(time.perf_counter() - started, 1)

    rows = [{"direction": label, "year": "all", **metrics(test, pred, score)}]
    for year in sorted(test["review_year"].unique()):
        mask = (test["review_year"] == year).to_numpy()
        if mask.sum() < 200:
            continue
        rows.append({"direction": label, "year": int(year), **metrics(test.loc[mask], pred[mask], score[mask])})
    overall = {"direction": label, "seconds": seconds, "train_n": len(train), "test_n": len(test)}
    return rows, overall


def plot(metrics_frame: pd.DataFrame, path) -> None:
    annual = metrics_frame.loc[metrics_frame["year"] != "all"].copy()
    annual["year"] = annual["year"].astype(int)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.84))
    for direction, part in annual.groupby("direction"):
        part = part.sort_values("year")
        axes[0].plot(part["year"], part["macro_f1"], marker="o", label=str(direction))
        axes[1].plot(part["year"], part["pr_auc"], marker="o", label=str(direction))
    axes[0].text(0.01, 1.02, "(a)", transform=axes[0].transAxes, fontsize=9, fontweight="bold")
    axes[0].set_xlabel("评论年份")
    axes[0].set_ylabel("宏 F1")
    axes[0].legend(fontsize=8)
    axes[1].text(0.01, 1.02, "(b)", transform=axes[1].transAxes, fontsize=9, fontweight="bold")
    axes[1].set_xlabel("评论年份")
    axes[1].set_ylabel("PR-AUC")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.xaxis.set_major_locator(MultipleLocator(4))
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="按年份回测文本模型")
    parser.add_argument("--limit", type=int, default=300000)
    args = parser.parse_args()
    setup_fonts()
    frame = load(args.limit)
    print(f"样本 {len(frame):,}，年份 {frame.review_year.min()}–{frame.review_year.max()}")

    forward, forward_summary = run_direction(frame, frame["review_year"] <= 2014, "2014及以前→2015+")
    backward, backward_summary = run_direction(frame, frame["review_year"] >= 2015, "2015+→2014及以前")
    metrics_frame = pd.DataFrame(forward + backward)
    metrics_frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    plot(metrics_frame, OUT_FIG)

    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED,
        "limit": args.limit,
        "n_rows": int(len(frame)),
        "forward": forward_summary,
        "backward": backward_summary,
        "annual": metrics_frame.to_dict("records"),
        "outputs": {"csv": str(OUT_CSV), "figure": str(OUT_FIG)},
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics_frame.to_string(index=False))
    print(f"写入 {OUT_JSON}")


if __name__ == "__main__":
    main()
