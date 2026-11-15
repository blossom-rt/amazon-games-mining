"""词级解释：TF-IDF + 逻辑回归的正负向 n-gram。

抽样 30 万条 in_scope 评论，按标签分层拆 80/20；向量化与模型都只在训练部分拟合，
验证部分只做评估。停用词保留否定词（not/never/no/n't 等），以便看到
"not worth"、"not good" 这类短语。

输出：
    results/tables/classify_terms.csv     正负向 top 词/短语
    results/tables/classify_terms.json    评估、否定/转折错误率、元数据
    results/figures/fig41_top_terms.png   正负向 top 15 系数

用法：
    python src/classify_terms.py
    python src/classify_terms.py --sample 100000
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from .classify import MAX_TEXT_CHARS
    from .config import FEATURES_PARQUET, FIGURES, FONT_CANDIDATES, SEED, TABLES
except ImportError:
    from classify import MAX_TEXT_CHARS
    from config import FEATURES_PARQUET, FIGURES, FONT_CANDIDATES, SEED, TABLES

OUT_TERMS = TABLES / "classify_terms.csv"
OUT_JSON = TABLES / "classify_terms.json"
OUT_FIG = FIGURES / "fig41_top_terms.png"
NEGATORS = {"not", "no", "nor", "never", "n't", "cannot"}
NEGATION_RE = re.compile(r"\b(?:not|never|no|n't|cannot|without)\b", re.IGNORECASE)
CONCESSION_RE = re.compile(r"\b(?:but|however|although|though|yet)\b", re.IGNORECASE)


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False


def hardware() -> dict:
    memory = None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        memory = round(status.ullTotalPhys / 2**30, 1)
    except Exception:
        pass
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_gb": memory,
        "python": platform.python_version(),
    }


def load_sample(sample: int) -> pd.DataFrame:
    frame = pq.read_table(
        FEATURES_PARQUET,
        columns=["asin", "label", "in_scope", "reviewText", "summary"],
    ).to_pandas()
    frame = frame.loc[frame["in_scope"]].reset_index(drop=True)
    if sample < len(frame):
        frame = frame.sample(n=sample, random_state=SEED).reset_index(drop=True)
    frame["text"] = (
        frame["reviewText"].fillna("") + " " + frame["summary"].fillna("")
    ).str.slice(0, MAX_TEXT_CHARS)
    assert frame["label"].nunique() == 2
    return frame


def top_terms(model: LogisticRegression, names: np.ndarray, count: int = 30) -> pd.DataFrame:
    coefficients = model.coef_[0]
    order = np.argsort(coefficients)
    rows = []
    for index in list(order[:count]) + list(order[-count:][::-1]):
        rows.append(
            {
                "term": str(names[index]),
                "coefficient": round(float(coefficients[index]), 4),
                "direction": "positive" if coefficients[index] > 0 else "negative",
            }
        )
    return pd.DataFrame(rows)


def plot_terms(terms: pd.DataFrame, path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for axis, direction, color, title in [
        (axes[0], "negative", "#c44e52", "负向（更可能 1–3 星）"),
        (axes[1], "positive", "#55a868", "正向（更可能 4–5 星）"),
    ]:
        part = terms.loc[terms["direction"] == direction].head(15).iloc[::-1]
        axis.barh(part["term"], part["coefficient"], color=color, alpha=0.85)
        axis.axvline(0, color="#333333", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("逻辑回归系数")
        axis.tick_params(axis="y", labelsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def error_rates(frame: pd.DataFrame, pred: np.ndarray) -> dict:
    text = frame["text"].to_numpy()
    y = frame["label"].to_numpy()
    negation = np.array([bool(NEGATION_RE.search(item)) for item in text])
    concession = np.array([bool(CONCESSION_RE.search(item)) for item in text])
    result = {}
    for name, mask in [("negation", negation), ("concession", concession)]:
        for value, label in [(True, "with"), (False, "without")]:
            subset = mask == value
            result[f"{name}_{label}"] = {
                "n": int(subset.sum()),
                "error_rate": round(float((pred[subset] != y[subset]).mean()), 4),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="词级解释")
    parser.add_argument("--sample", type=int, default=300_000)
    args = parser.parse_args()
    setup_fonts()
    started = time.perf_counter()

    frame = load_sample(args.sample)
    y = frame["label"].to_numpy(dtype=np.int8)
    train_index, valid_index = train_test_split(
        np.arange(len(frame)), test_size=0.2, random_state=SEED, stratify=y
    )
    stop_words = sorted(set(ENGLISH_STOP_WORDS) - NEGATORS)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=10,
        max_features=80_000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w[\w']+\b",
        stop_words=stop_words,
        lowercase=True,
        dtype=np.float32,
    )
    train_matrix = vectorizer.fit_transform(frame["text"].iloc[train_index])
    valid_matrix = vectorizer.transform(frame["text"].iloc[valid_index])
    names = vectorizer.get_feature_names_out()
    print(f"样本 {len(frame):,}；词表 {len(names):,}；训练矩阵 {train_matrix.shape}")

    model = LogisticRegression(
        solver="liblinear", C=1.0, max_iter=300, random_state=SEED
    )
    model.fit(train_matrix, y[train_index])
    score = model.predict_proba(valid_matrix)[:, 1]
    pred = (score >= 0.5).astype(np.int8)
    evaluation = {
        "accuracy": round(float(accuracy_score(y[valid_index], pred)), 4),
        "macro_f1": round(float(f1_score(y[valid_index], pred, average="macro")), 4),
        "pr_auc": round(float(average_precision_score(y[valid_index], score)), 4),
        "roc_auc": round(float(roc_auc_score(y[valid_index], score)), 4),
    }

    terms = top_terms(model, names, count=30)
    doc_freq = np.asarray((train_matrix > 0).sum(axis=0)).ravel()
    term_index = {name: index for index, name in enumerate(names)}
    terms["train_documents"] = [int(doc_freq[term_index[name]]) for name in terms["term"]]
    terms.to_csv(OUT_TERMS, index=False, encoding="utf-8-sig")

    # 只对验证集做否定/转折的错误率，避免把训练拟合当成结论
    valid_frame = frame.iloc[valid_index].reset_index(drop=True)
    valid_pred = pred
    errors = error_rates(valid_frame, valid_pred)

    plot_terms(terms, OUT_FIG)
    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED,
        "sample": int(len(frame)),
        "train": int(len(train_index)),
        "valid": int(len(valid_index)),
        "vocabulary": int(len(names)),
        "evaluation": evaluation,
        "error_rates": errors,
        "top_positive": terms.loc[terms["direction"] == "positive"].head(20).to_dict("records"),
        "top_negative": terms.loc[terms["direction"] == "negative"].head(20).to_dict("records"),
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "hardware": hardware(),
        "outputs": {"terms": str(OUT_TERMS), "figure": str(OUT_FIG)},
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["sample", "vocabulary", "evaluation", "error_rates"]},
                     ensure_ascii=False, indent=2))
    print(terms.head(12).to_string(index=False))
    print(f"写入 {OUT_JSON}")


if __name__ == "__main__":
    main()
