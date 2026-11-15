"""否定/转折专项：否定范围特征与 error-focused 评估。

做法：
- 在原文上给否定词后的 3 个词加 `neg_` 前缀（简单否定范围标记）；
- 与原始 TF-IDF 在同一 5 折、同一线性 SVM 上对比；
- 分别统计“含否定词/含转折词”子集的错误率，而不是只看总体宏 F1。

用法：
    python src/classify_negation.py
    python src/classify_negation.py --limit 150000
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

try:
    from .classify import MAX_TEXT_CHARS, N_HASH
    from .config import FEATURES_PARQUET, FIGURES, FONT_CANDIDATES, SEED, TABLES
except ImportError:
    from classify import MAX_TEXT_CHARS, N_HASH
    from config import FEATURES_PARQUET, FIGURES, FONT_CANDIDATES, SEED, TABLES

OUT_CSV = TABLES / "classify_negation_metrics.csv"
OUT_JSON = TABLES / "classify_negation.json"
OUT_FIG = FIGURES / "fig44_negation_scope.png"

NEGATORS = {
    "not", "no", "never", "none", "nothing", "neither", "nor", "hardly",
    "barely", "scarcely", "without", "cannot", "cant", "dont", "doesnt",
    "didnt", "isnt", "arent", "wasnt", "werent", "wont", "wouldnt",
    "couldnt", "shouldnt", "aint",
}
CONCESSIONS = {"but", "however", "although", "though", "yet", "still", "despite"}
TOKEN_RE = re.compile(r"[A-Za-z']+|[.!?;]")


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False


def normalise(token: str) -> str:
    return token.lower().replace("'", "")


def negate_scope(text: str) -> str:
    """给否定词后的 3 个实词加 neg_ 前缀，遇到标点/转折词提前结束。"""
    tokens = TOKEN_RE.findall(text.lower())
    output = []
    remaining = 0
    for token in tokens:
        if token in ".!?;":
            remaining = 0
            output.append(token)
            continue
        if normalise(token) in CONCESSIONS:
            remaining = 0
            output.append(token)
            continue
        if normalise(token) in NEGATORS:
            remaining = 3
            output.append(token)
            continue
        if remaining > 0 and token.isalpha():
            output.append("neg_" + token)
            remaining -= 1
        else:
            output.append(token)
    return " ".join(output)


def load(limit: int) -> pd.DataFrame:
    frame = pq.read_table(
        FEATURES_PARQUET,
        columns=["asin", "label", "in_scope", "reviewText", "summary", "fold"],
    ).to_pandas()
    frame = frame.loc[frame["in_scope"]].reset_index(drop=True)
    if limit < len(frame):
        frame = frame.sample(n=limit, random_state=SEED).reset_index(drop=True)
    assert frame["fold"].between(0, 4).all(), "折号不完整"
    return frame


def build_matrix(text: np.ndarray, train_index, valid_index):
    vectorizer = HashingVectorizer(
        n_features=N_HASH, alternate_sign=False, norm=None, dtype=np.float32,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    counts = vectorizer.transform(text)
    tfidf = TfidfTransformer(sublinear_tf=True)
    x_train = tfidf.fit_transform(counts[train_index])
    x_valid = tfidf.transform(counts[valid_index])
    model = SGDClassifier(
        loss="hinge", alpha=1e-5, max_iter=15, tol=1e-3,
        class_weight="balanced", random_state=SEED,
    )
    model.fit(x_train, labels[train_index])
    score = model.decision_function(x_valid)
    pred = (score >= 0).astype(np.int8)
    return pred, score


def main() -> None:
    parser = argparse.ArgumentParser(description="否定范围特征与错误率对比")
    parser.add_argument("--limit", type=int, default=150000)
    args = parser.parse_args()
    setup_fonts()
    started = time.perf_counter()
    frame = load(args.limit)
    global labels
    labels = frame["label"].to_numpy(dtype=np.int8)
    raw_text = (frame["reviewText"].fillna("") + " " + frame["summary"].fillna("")) \
        .str.slice(0, MAX_TEXT_CHARS).to_numpy()
    scoped_text = np.array([negate_scope(value) for value in raw_text], dtype=object)

    has_neg = np.array([any(normalise(t) in NEGATORS for t in TOKEN_RE.findall(x.lower())) for x in raw_text])
    has_con = np.array([any(normalise(t) in CONCESSIONS for t in TOKEN_RE.findall(x.lower())) for x in raw_text])

    variants = {"raw": raw_text, "neg_scope": scoped_text}
    oof = {name: {"pred": np.full(len(frame), -1, dtype=np.int8), "score": np.full(len(frame), np.nan)} for name in variants}
    rows = []
    for name, values in variants.items():
        for fold in range(5):
            train_index = np.flatnonzero(frame["fold"].to_numpy() != fold)
            valid_index = np.flatnonzero(frame["fold"].to_numpy() == fold)
            pred, score = build_matrix(values, train_index, valid_index)
            oof[name]["pred"][valid_index] = pred
            oof[name]["score"][valid_index] = score
            y = labels[valid_index]
            rows.append({
                "variant": name, "fold": fold + 1, "n": len(y),
                "accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro")),
                "pr_auc": float(average_precision_score(y, score)),
                "roc_auc": float(roc_auc_score(y, score)),
            })
        print(f"{name} 完成")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    summary = {}
    for name in variants:
        part = metrics.loc[metrics["variant"] == name]
        pred = oof[name]["pred"]
        error = pred != labels
        summary[name] = {
            "macro_f1": {"mean": round(float(part["macro_f1"].mean()), 4), "std": round(float(part["macro_f1"].std(ddof=1)), 4)},
            "pr_auc": {"mean": round(float(part["pr_auc"].mean()), 4), "std": round(float(part["pr_auc"].std(ddof=1)), 4)},
            "roc_auc": {"mean": round(float(part["roc_auc"].mean()), 4), "std": round(float(part["roc_auc"].std(ddof=1)), 4)},
            "accuracy": round(float(accuracy_score(labels, pred)), 4),
            "error_rate": round(float(error.mean()), 4),
            "error_rate_negation": round(float(error[has_neg].mean()), 4),
            "error_rate_concession": round(float(error[has_con].mean()), 4),
            "n_negation": int(has_neg.sum()),
            "n_concession": int(has_con.sum()),
        }

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    names = list(variants)
    means = [summary[name]["macro_f1"]["mean"] for name in names]
    stds = [summary[name]["macro_f1"]["std"] for name in names]
    axes[0].bar(names, means, yerr=stds, capsize=4, color=["#4c72b0", "#dd8452"])
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_ylabel("宏 F1")
    axes[0].set_title("否定范围特征：宏 F1")
    width = 0.35
    x = np.arange(2)
    axes[1].bar(x - width / 2, [summary[name]["error_rate_negation"] for name in names], width, label="含否定词", color="#c44e52")
    axes[1].bar(x + width / 2, [summary[name]["error_rate_concession"] for name in names], width, label="含转折词", color="#8172b2")
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("错误率")
    axes[1].set_title("否定/转折子集错误率")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(figure)

    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED,
        "limit": args.limit,
        "n_rows": int(len(frame)),
        "positive_rate": round(float(labels.mean()), 4),
        "variants": summary,
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "outputs": {"csv": str(OUT_CSV), "figure": str(OUT_FIG)},
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"写入 {OUT_JSON}")


if __name__ == "__main__":
    main()
