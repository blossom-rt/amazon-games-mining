"""P4 稳健性检查：10 折分组评估、效应量、校准与业务阈值。

与 classify.py 的区别：
- 只训练 Dummy、逻辑回归（SGD log_loss）与线性 SVM（SGD hinge），
  用 10 折而不是 5 折，提高 Wilcoxon 检验的最小可达 p 值；
- 报告配对差值、Cohen's d_z、rank-biserial 与 bootstrap 置信区间；
- 对逻辑回归的 OOF 概率做分组 isotonic 校准，报告 Brier、log loss、ECE；
- 在校准后的概率上给 F1 最优阈值与代价敏感阈值（漏判成本 5 倍假阳性）。

用法：
    python src/classify_robustness.py
    python src/classify_robustness.py --limit 20000
"""

from __future__ import annotations

import argparse
import gc
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
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import rankdata, wilcoxon

try:
    from .classify import CAT_COLS, MAX_TEXT_CHARS, MODEL_FEATURES, N_HASH
    from .config import FEATURES_PARQUET, FIGURES, FONT_CANDIDATES, SEED, TABLES
    from .classify import model_and_score
except ImportError:
    from classify import CAT_COLS, MAX_TEXT_CHARS, MODEL_FEATURES, N_HASH
    from config import FEATURES_PARQUET, FIGURES, FONT_CANDIDATES, SEED, TABLES
    from classify import model_and_score

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

N_FOLDS = 10
CALIBRATION_BINS = 10
N_BOOTSTRAP = 10_000
FN_COST = 5.0
FP_COST = 1.0
METRIC_NAMES = ["accuracy", "macro_f1", "pr_auc", "roc_auc"]
OUT_METRICS = TABLES / "classify_10fold_metrics.csv"
OUT_JSON = TABLES / "classify_robustness.json"
OUT_CALIBRATION = TABLES / "classify_calibration.csv"
OUT_CAL_FIG = FIGURES / "fig36_calibration.png"
OUT_THRESHOLD_FIG = FIGURES / "fig37_threshold.png"


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False


def load_frame(limit: int | None) -> pd.DataFrame:
    columns = [
        "asin", "label", "in_scope", "reviewText", "summary",
    ] + MODEL_FEATURES
    frame = pq.read_table(FEATURES_PARQUET, columns=columns).to_pandas()
    frame = frame.loc[frame["in_scope"]].reset_index(drop=True)
    if limit is not None and limit < len(frame):
        frame = frame.sample(n=limit, random_state=SEED).reset_index(drop=True)
    assert len(frame) >= 10_000, "样本过少，无法做 10 折"
    return frame


def make_folds(frame: pd.DataFrame) -> np.ndarray:
    splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    )
    folds = np.full(len(frame), -1, dtype=np.int8)
    for index, (_, valid) in enumerate(
        splitter.split(frame, frame["label"], groups=frame["asin"])
    ):
        folds[valid] = index
    assert (folds >= 0).all(), "存在未分配折的样本"
    assert folds.max() == N_FOLDS - 1, "折数不完整"
    return folds


def ece(y_true: np.ndarray, prob: np.ndarray, bins: int = CALIBRATION_BINS) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (prob >= low) & (prob < high if high < 1 else prob <= high)
        if not mask.any():
            continue
        total += mask.mean() * abs(y_true[mask].mean() - prob[mask].mean())
    return float(total)


def calibration_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(y_true, prob)),
        "log_loss": float(log_loss(y_true, prob)),
        "ece": ece(y_true, prob),
    }


def grouped_isotonic(
    y_true: np.ndarray, prob: np.ndarray, folds: np.ndarray
) -> np.ndarray:
    calibrated = np.full(len(prob), np.nan, dtype=np.float64)
    for fold in range(N_FOLDS):
        train = folds != fold
        valid = folds == fold
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(prob[train], y_true[train])
        calibrated[valid] = model.predict(prob[valid])
    assert not np.isnan(calibrated).any(), "校准结果存在缺失"
    return np.clip(calibrated, 0.0, 1.0)


def paired_stats(diff: np.ndarray) -> dict:
    diff = np.asarray(diff, dtype=float)
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = wilcoxon(diff)
    ranks = rankdata(np.abs(diff))
    rank_biserial = float(
        (ranks[diff > 0].sum() - ranks[diff < 0].sum()) / ranks.sum()
    ) if diff.any() else 0.0
    std = float(diff.std(ddof=1))
    cohen_dz = float(diff.mean() / std) if std > 0 else 0.0
    rng = np.random.default_rng(SEED)
    means = np.array(
        [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(N_BOOTSTRAP)]
    )
    return {
        "n_pairs": int(len(diff)),
        "mean_difference": float(diff.mean()),
        "std_difference": std,
        "cohen_dz": cohen_dz,
        "rank_biserial": rank_biserial,
        "wilcoxon_statistic": float(statistic),
        "p_value": float(p_value),
        "bootstrap_ci95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }


def choose_thresholds(y_true: np.ndarray, prob: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_index = int(np.nanargmax(f1[:-1]))
    best_threshold = float(thresholds[best_index])

    grid = np.linspace(0.05, 0.95, 91)
    costs = []
    for threshold in grid:
        pred = prob >= threshold
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        costs.append(FN_COST * fn + FP_COST * fp)
    cost_index = int(np.argmin(costs))
    cost_threshold = float(grid[cost_index])
    return {
        "f1_optimal": {
            "threshold": round(best_threshold, 4),
            "precision": round(float(precision[best_index]), 4),
            "recall": round(float(recall[best_index]), 4),
            "f1": round(float(f1[best_index]), 4),
        },
        "cost_sensitive": {
            "threshold": round(cost_threshold, 4),
            "fn_cost": FN_COST,
            "fp_cost": FP_COST,
            "total_cost": float(costs[cost_index]),
        },
    }


def plot_calibration(y_true, raw, calibrated, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for axis, prob, label in [
        (axes[0], raw, "逻辑回归原始概率"),
        (axes[0], calibrated, "isotonic 校准后"),
    ]:
        frac, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
        axis.plot(mean_pred, frac, marker="o", linewidth=1.4, label=label)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    axes[0].set_xlabel("预测概率")
    axes[0].set_ylabel("实际正类比例")
    axes[0].set_title("校准曲线（OOF）")
    axes[0].legend(fontsize=8)

    axes[1].hist(raw, bins=40, alpha=0.55, label="原始", color="#4c72b0")
    axes[1].hist(calibrated, bins=40, alpha=0.55, label="校准后", color="#dd8452")
    axes[1].set_xlabel("预测概率")
    axes[1].set_ylabel("样本数")
    axes[1].set_title("概率分布")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_threshold(y_true, prob, thresholds: dict, path: Path) -> None:
    grid = np.linspace(0.05, 0.95, 91)
    f1 = []
    for threshold in grid:
        pred = (prob >= threshold).astype(np.int8)
        f1.append(f1_score(y_true, pred, average="macro", zero_division=0))
    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    axis.plot(grid, f1, color="#4c72b0", linewidth=1.6)
    axis.axvline(thresholds["f1_optimal"]["threshold"], color="#55a868", linestyle="--",
                 label=f"F1 最优 {thresholds['f1_optimal']['threshold']:.2f}")
    axis.axvline(thresholds["cost_sensitive"]["threshold"], color="#c44e52", linestyle="--",
                 label=f"代价最优 {thresholds['cost_sensitive']['threshold']:.2f}")
    axis.set_xlabel("阈值")
    axis.set_ylabel("宏 F1")
    axis.set_title("业务阈值选择（校准后 OOF）")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="10 折稳健性、校准与业务阈值")
    parser.add_argument("--limit", type=int, default=None, help="调试抽样行数")
    args = parser.parse_args()

    setup_fonts()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    frame = load_frame(args.limit)
    folds = make_folds(frame)
    labels = frame["label"].to_numpy(dtype=np.int8)
    print(f"样本 {len(frame):,}，正类 {labels.mean():.4f}，{N_FOLDS} 折")

    text = (
        frame["reviewText"].fillna("") + " " + frame["summary"].fillna("")
    ).str.slice(0, MAX_TEXT_CHARS).to_numpy()
    categories = frame[CAT_COLS].fillna("(missing)").astype(str)
    numbers = frame[MODEL_FEATURES].drop(columns=CAT_COLS).astype("float32")

    vectorizer = HashingVectorizer(
        n_features=N_HASH,
        alternate_sign=False,
        norm=None,
        dtype=np.float32,
        token_pattern=r"(?u)\b\w\w+\b",
        lowercase=True,
    )
    print("文本 hashing")
    counts = vectorizer.transform(text)
    del text
    gc.collect()
    print(f"文本矩阵 {counts.shape[0]:,} × {counts.shape[1]:,}")

    names = ["dummy", "logistic_sgd", "linear_svm"]
    scores = {name: np.full(len(frame), np.nan) for name in names}
    preds = {name: np.full(len(frame), -1, dtype=np.int8) for name in names}
    rows: list[dict] = []

    for fold in range(N_FOLDS):
        fold_started = time.perf_counter()
        train = folds != fold
        valid = folds == fold
        y_train, y_valid = labels[train], labels[valid]
        train_index, valid_index = np.flatnonzero(train), np.flatnonzero(valid)
        print(f"第 {fold + 1}/{N_FOLDS} 折：训练 {len(y_train):,}，验证 {len(y_valid):,}")

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

        tfidf = TfidfTransformer(sublinear_tf=True)
        text_train = tfidf.fit_transform(counts[train_index])
        text_valid = tfidf.transform(counts[valid_index])

        x_train = sp.hstack(
            [sp.csr_matrix(number_train), category_train, text_train], format="csr"
        )
        x_valid = sp.hstack(
            [sp.csr_matrix(number_valid), category_valid, text_valid], format="csr"
        )

        for name in names:
            model_started = time.perf_counter()
            _, score, pred = model_and_score(name, x_train, y_train, x_valid)
            seconds = time.perf_counter() - model_started
            scores[name][valid_index] = score
            preds[name][valid_index] = pred
            rows.append(
                {
                    "model": name,
                    "fold": fold + 1,
                    "n_train": len(y_train),
                    "n_valid": len(y_valid),
                    "accuracy": float(accuracy_score(y_valid, pred)),
                    "macro_f1": float(f1_score(y_valid, pred, average="macro")),
                    "pr_auc": float(average_precision_score(y_valid, score)),
                    "roc_auc": float(roc_auc_score(y_valid, score)),
                    "seconds": round(seconds, 2),
                }
            )
        del x_train, x_valid, text_train, text_valid
        del category_train, category_valid, number_train, number_valid
        gc.collect()
        print(f"  折用时 {time.perf_counter() - fold_started:.1f} 秒")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")

    summary = {}
    for name in names:
        part = metrics.loc[metrics["model"] == name]
        summary[name] = {
            metric: {
                "mean": round(float(part[metric].mean()), 4),
                "std": round(float(part[metric].std(ddof=1)), 4),
            }
            for metric in METRIC_NAMES
        }
        summary[name]["seconds"] = round(float(part["seconds"].mean()), 1)

    best = max(names, key=lambda name: summary[name]["macro_f1"]["mean"])
    dummy_f1 = metrics.loc[metrics["model"] == "dummy", "macro_f1"].to_numpy()
    best_f1 = metrics.loc[metrics["model"] == best, "macro_f1"].to_numpy()
    test = paired_stats(best_f1 - dummy_f1)
    test["best_model"] = best
    test["comparison"] = f"{best} - dummy"

    lr_test = None
    lr_f1 = metrics.loc[metrics["model"] == "logistic_sgd", "macro_f1"].to_numpy()
    svm_f1 = metrics.loc[metrics["model"] == "linear_svm", "macro_f1"].to_numpy()
    if best == "linear_svm" and not np.allclose(svm_f1, lr_f1):
        lr_test = paired_stats(svm_f1 - lr_f1)
        lr_test["comparison"] = "linear_svm - logistic_sgd"

    raw_prob = np.clip(scores["logistic_sgd"], 1e-6, 1 - 1e-6)
    calibrated = grouped_isotonic(labels, raw_prob, folds)
    raw_metrics = calibration_metrics(labels, raw_prob)
    cal_metrics = calibration_metrics(labels, calibrated)
    thresholds = choose_thresholds(labels, calibrated)

    calibration_rows = []
    for label, prob in [("raw", raw_prob), ("isotonic", calibrated)]:
        row = {"calibration": label, **calibration_metrics(labels, prob)}
        calibration_rows.append(row)
    pd.DataFrame(calibration_rows).to_csv(OUT_CALIBRATION, index=False, encoding="utf-8-sig")

    plot_calibration(labels, raw_prob, calibrated, OUT_CAL_FIG)
    plot_threshold(labels, calibrated, thresholds, OUT_THRESHOLD_FIG)

    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED,
        "n_folds": N_FOLDS,
        "n_rows": int(len(frame)),
        "positive_rate": round(float(labels.mean()), 4),
        "fold_sizes": [int((folds == fold).sum()) for fold in range(N_FOLDS)],
        "models": summary,
        "best_model": best,
        "primary_test": test,
        "secondary_test": lr_test,
        "calibration": {
            "raw": raw_metrics,
            "isotonic": cal_metrics,
            "method": "grouped 10-fold isotonic，校准器只在其余折的 OOF 概率上拟合",
        },
        "thresholds": thresholds,
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "hardware": hardware(),
        "outputs": {
            "metrics": str(OUT_METRICS),
            "calibration": str(OUT_CALIBRATION),
            "fig_calibration": str(OUT_CAL_FIG),
            "fig_threshold": str(OUT_THRESHOLD_FIG),
        },
    }
    OUT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: result[k] for k in [
        "n_rows", "positive_rate", "best_model", "primary_test",
        "calibration", "thresholds", "runtime_seconds",
    ]}, ensure_ascii=False, indent=2))
    print(f"写入 {OUT_JSON}")


if __name__ == "__main__":
    main()
