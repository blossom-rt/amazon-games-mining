"""P4 模态消融：文本、元数据、合并三种输入的贡献。

同一批 in_scope 评论、同一套 5 折（沿用 features.parquet 的 fold 列）、
同一预处理与线性 SVM，只改变输入矩阵：

- text_only：Hashing + 每折 TF-IDF；
- metadata_only：数值 + 类别，不含评论文本；
- combined：文本 + 元数据，与 classify.py 主模型一致。

Dummy 只在 combined 上训练一次，作为统一基线。输出
results/tables/classify_ablation_metrics.csv、classify_ablation.json、
results/figures/fig40_ablation.png。

用法：
    python src/classify_ablation.py
    python src/classify_ablation.py --limit 50000
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .classify import (
        CAT_COLS,
        MAX_TEXT_CHARS,
        MODEL_FEATURES,
        N_HASH,
        build_text,
        model_and_score,
    )
    from .config import FEATURES_PARQUET, FIGURES, FONT_CANDIDATES, SEED, TABLES
except ImportError:
    from classify import (
        CAT_COLS,
        MAX_TEXT_CHARS,
        MODEL_FEATURES,
        N_HASH,
        build_text,
        model_and_score,
    )
    from config import FEATURES_PARQUET, FIGURES, FONT_CANDIDATES, SEED, TABLES

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

N_FOLDS = 5
OUT_METRICS = TABLES / "classify_ablation_metrics.csv"
OUT_JSON = TABLES / "classify_ablation.json"
OUT_FIG = FIGURES / "fig40_ablation.png"
VARIANTS = ["text_only", "metadata_only", "combined"]


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


def plot_ablation(summary: dict, path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    labels = ["文本", "元数据", "合并"]
    keys = ["text_only", "metadata_only", "combined"]
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    x = np.arange(len(labels))
    width = 0.36
    for axis, metric, title in [
        (axes[0], "macro_f1", "宏 F1"),
        (axes[1], "pr_auc", "PR-AUC"),
    ]:
        means = [summary[key][metric]["mean"] for key in keys]
        stds = [summary[key][metric]["std"] for key in keys]
        axis.bar(x, means, width=0.55, color=colors, alpha=0.85)
        axis.errorbar(x, means, yerr=stds, fmt="none", ecolor="#333333", capsize=4)
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.set_ylabel(title)
        axis.set_ylim(0.5 if metric == "macro_f1" else 0.8, 1.0)
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate(means):
            axis.text(index, value + 0.005, f"{value:.4f}", ha="center", fontsize=8)
    axes[0].set_title("模态消融：宏 F1")
    axes[1].set_title("模态消融：PR-AUC")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="文本/元数据消融")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    setup_fonts()
    started = time.perf_counter()

    columns = ["asin", "label", "fold", "in_scope", "reviewText", "summary"] + MODEL_FEATURES
    frame = pq.read_table(FEATURES_PARQUET, columns=columns).to_pandas()
    frame = frame.loc[frame["in_scope"]].reset_index(drop=True)
    if args.limit is not None and args.limit < len(frame):
        frame = frame.sample(n=args.limit, random_state=SEED).reset_index(drop=True)
    assert len(frame) >= 20_000

    labels = frame["label"].to_numpy(dtype=np.int8)
    folds = frame["fold"].to_numpy()
    if len(np.unique(folds)) > N_FOLDS:
        folds = folds % N_FOLDS
    assert folds.min() >= 0 and folds.max() < N_FOLDS

    categories = frame[CAT_COLS].fillna("(missing)").astype(str)
    numbers = frame[MODEL_FEATURES].drop(columns=CAT_COLS).astype("float32")
    text = build_text(frame)
    vectorizer = HashingVectorizer(
        n_features=N_HASH,
        alternate_sign=False,
        norm=None,
        dtype=np.float32,
        token_pattern=r"(?u)\b\w\w+\b",
        lowercase=True,
    )
    counts = vectorizer.transform(text)
    del text
    gc.collect()
    print(f"样本 {len(frame):,}，正类 {labels.mean():.4f}，文本矩阵 {counts.shape}")

    rows: list[dict] = []
    for fold in range(N_FOLDS):
        train = folds != fold
        valid = folds == fold
        y_train, y_valid = labels[train], labels[valid]
        train_index, valid_index = np.flatnonzero(train), np.flatnonzero(valid)
        print(f"第 {fold + 1}/{N_FOLDS} 折")

        imputer = SimpleImputer(strategy="median", add_indicator=True)
        number_train = imputer.fit_transform(numbers.iloc[train_index])
        number_valid = imputer.transform(numbers.iloc[valid_index])
        scaler = StandardScaler()
        number_train = scaler.fit_transform(number_train).astype(np.float32)
        number_valid = scaler.transform(number_valid).astype(np.float32)
        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=50,
            sparse_output=True,
            dtype=np.float32,
        )
        category_train = encoder.fit_transform(categories.iloc[train_index])
        category_valid = encoder.transform(categories.iloc[valid_index])
        meta_train = sp.hstack([sp.csr_matrix(number_train), category_train], format="csr")
        meta_valid = sp.hstack([sp.csr_matrix(number_valid), category_valid], format="csr")

        tfidf = TfidfTransformer(sublinear_tf=True)
        text_train = tfidf.fit_transform(counts[train_index])
        text_valid = tfidf.transform(counts[valid_index])
        combined_train = sp.hstack([meta_train, text_train], format="csr")
        combined_valid = sp.hstack([meta_valid, text_valid], format="csr")

        matrices = {
            "text_only": (text_train, text_valid),
            "metadata_only": (meta_train, meta_valid),
            "combined": (combined_train, combined_valid),
        }

        _, score, pred = model_and_score("dummy", combined_train, y_train, combined_valid)
        rows.append(_row("dummy", fold, y_valid, score, pred, 0.0, len(y_train), len(y_valid)))

        for name in VARIANTS:
            x_train, x_valid = matrices[name]
            model_started = time.perf_counter()
            _, score, pred = model_and_score("linear_svm", x_train, y_train, x_valid)
            seconds = time.perf_counter() - model_started
            rows.append(_row(name, fold, y_valid, score, pred, seconds, len(y_train), len(y_valid)))
            print(f"  {name}: 宏F1 {rows[-1]['macro_f1']:.4f}，{seconds:.1f}s")

        del matrices, combined_train, combined_valid, text_train, text_valid
        del meta_train, meta_valid, category_train, category_valid
        del number_train, number_valid, tfidf
        gc.collect()

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")

    summary = {}
    for name in ["dummy"] + VARIANTS:
        part = metrics.loc[metrics["model"] == name]
        summary[name] = {
            metric: {
                "mean": round(float(part[metric].mean()), 4),
                "std": round(float(part[metric].std(ddof=1)), 4),
            }
            for metric in ["accuracy", "macro_f1", "pr_auc", "roc_auc"]
        }
        summary[name]["seconds"] = round(float(part["seconds"].mean()), 1)

    combined_f1 = metrics.loc[metrics["model"] == "combined", "macro_f1"].to_numpy()
    deltas = {}
    for name in ["text_only", "metadata_only"]:
        values = metrics.loc[metrics["model"] == name, "macro_f1"].to_numpy()
        deltas[name] = {
            "mean_delta_vs_combined": round(float((values - combined_f1).mean()), 4),
            "per_fold": [round(float(x), 4) for x in (values - combined_f1)],
        }

    plot_ablation(summary, OUT_FIG)
    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED,
        "n_folds": N_FOLDS,
        "n_rows": int(len(frame)),
        "positive_rate": round(float(labels.mean()), 4),
        "models": summary,
        "deltas_vs_combined": deltas,
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "hardware": hardware(),
        "outputs": {"metrics": str(OUT_METRICS), "figure": str(OUT_FIG)},
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["n_rows", "models", "deltas_vs_combined", "runtime_seconds"]},
                     ensure_ascii=False, indent=2))
    print(f"写入 {OUT_JSON}")


def _row(name, fold, y_true, score, pred, seconds, n_train, n_valid) -> dict:
    return {
        "model": name,
        "fold": fold + 1,
        "n_train": n_train,
        "n_valid": n_valid,
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "pr_auc": float(average_precision_score(y_true, score)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "seconds": round(float(seconds), 2),
    }


if __name__ == "__main__":
    main()
