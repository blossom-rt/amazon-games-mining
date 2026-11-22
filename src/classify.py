"""P4 评论正向/负向分类。

只用游戏本体评论，按已固定的 asin 分组 5 折训练与评估。
文本用 hashing 向量化，IDF、缺失填补、标准化、SVD 全部只在训练折拟合。
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

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from scipy.stats import wilcoxon
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    from .config import (
        DATA_PROCESSED,
        FEATURES_PARQUET,
        FIGURES,
        FONT_CANDIDATES,
        N_SPLITS,
        POSITIVE_RATING_THRESHOLD,
        SEED,
        TABLES,
    )
    from .build_features import MODEL_FEATURES
except ImportError:
    from config import (
        DATA_PROCESSED,
        FEATURES_PARQUET,
        FIGURES,
        FONT_CANDIDATES,
        N_SPLITS,
        POSITIVE_RATING_THRESHOLD,
        SEED,
        TABLES,
    )
    from build_features import MODEL_FEATURES


CAT_COLS = ["main_cat_clean", "category_leaf"]
NUMERIC_COLS = [name for name in MODEL_FEATURES if name not in CAT_COLS]
MODEL_NAMES = ["dummy", "logistic_sgd", "linear_svm", "hist_gb", "random_forest"]
MODEL_LABELS = {
    "dummy": "Dummy 基线",
    "logistic_sgd": "逻辑回归（SGD）",
    "linear_svm": "线性 SVM（SGD）",
    "hist_gb": "HistGradientBoosting",
    "random_forest": "随机森林",
}
MODEL_COLORS = {
    "dummy": "#8C8C8C",
    "logistic_sgd": "#DD8452",
    "linear_svm": "#4C72B0",
    "hist_gb": "#55A868",
    "random_forest": "#8172B3",
}
METRIC_NAMES = ["accuracy", "macro_f1", "pr_auc", "roc_auc"]

MAX_TEXT_CHARS = 3000
N_HASH = 2**18
SVD_COMPONENTS = 20
SVD_FIT_ROWS = 100_000
PERMUTATION_ROWS = 20_000
PERMUTATION_REPEATS = 5

ERROR_CSV = TABLES / "classify_errors.csv"
METRICS_CSV = TABLES / "classify_metrics.csv"
SUMMARY_JSON = TABLES / "classify_summary.json"
IMPORTANCE_CSV = TABLES / "classify_importance.csv"
SHAP_JSON = TABLES / "classify_shap.json"


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False


def hardware() -> dict:
    memory_gb = None
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
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

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            memory_gb = round(status.ullTotalPhys / 1024**3, 1)
        except Exception:
            memory_gb = None
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_gb": memory_gb,
        "python": platform.python_version(),
    }


def load_frame(limit: int | None) -> pd.DataFrame:
    columns = [
        "reviewerID",
        "asin",
        "label",
        "fold",
        "in_scope",
        "reviewText",
        "summary",
    ] + MODEL_FEATURES
    table = pq.read_table(FEATURES_PARQUET, columns=columns)
    frame = table.to_pandas()
    frame = frame.loc[frame["in_scope"]].reset_index(drop=True)
    if limit is not None and limit < len(frame):
        frame = frame.sample(n=limit, random_state=SEED).reset_index(drop=True)
        print(f"快速模式：抽取 {len(frame):,} 行，结果只用于调试")
    assert frame["fold"].between(0, N_SPLITS - 1).all(), "存在未分配折的样本"
    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES)), "建模特征列重复"
    return frame


def build_text(frame: pd.DataFrame) -> np.ndarray:
    text = (
        frame["reviewText"].fillna("")
        + " "
        + frame["summary"].fillna("")
    ).str.slice(0, MAX_TEXT_CHARS)
    return text.to_numpy()


def build_categories(frame: pd.DataFrame) -> pd.DataFrame:
    categories = frame[CAT_COLS].copy()
    for column in CAT_COLS:
        categories[column] = categories[column].fillna("(missing)").astype(str)
    return categories


def model_and_score(name: str, x_train, y_train, x_valid):
    if name == "dummy":
        model = DummyClassifier(strategy="prior")
        model.fit(np.zeros((len(y_train), 1), dtype=np.float32), y_train)
        dummy_valid = np.zeros((x_valid.shape[0], 1), dtype=np.float32)
        score = model.predict_proba(dummy_valid)[:, 1]
        pred = model.predict(dummy_valid)
        return model, score, pred

    if name == "logistic_sgd":
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-5,
            max_iter=8,
            tol=None,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
    elif name == "linear_svm":
        model = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=1e-5,
            max_iter=15,
            tol=1e-3,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
    elif name == "hist_gb":
        model = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=31,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=SEED,
        )
    elif name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=60,
            max_depth=18,
            min_samples_leaf=40,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        )
    else:
        raise ValueError(name)

    if name == "hist_gb":
        weights = compute_sample_weight("balanced", y_train)
        model.fit(x_train, y_train, sample_weight=weights)
    else:
        model.fit(x_train, y_train)

    if hasattr(model, "predict_proba"):
        score = model.predict_proba(x_valid)[:, 1]
    else:
        score = model.decision_function(x_valid)
    pred = model.predict(x_valid)
    return model, score, pred


def metric_row(name: str, fold: int, y_true, score, pred, seconds: float, n_train: int, n_valid: int) -> dict:
    return {
        "model": name,
        "model_label": MODEL_LABELS[name],
        "fold": fold,
        "n_train": n_train,
        "n_valid": n_valid,
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "seconds": round(seconds, 1),
    }


def plot_curves(y_true, scores: dict, roc_path: Path, pr_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 4.55))
    for name in MODEL_NAMES:
        score = scores[name]
        fpr, tpr, _ = roc_curve(y_true, score)
        axis.plot(
            fpr,
            tpr,
            linewidth=1.4,
            color=MODEL_COLORS[name],
            label=f"{MODEL_LABELS[name]} (AUC {roc_auc_score(y_true, score):.3f})",
        )
    axis.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    axis.set_xlabel("假正率 FPR")
    axis.set_ylabel("真正率 TPR")
    axis.legend(fontsize=7, loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(roc_path, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.2, 4.55))
    for name in MODEL_NAMES:
        score = scores[name]
        precision, recall, _ = precision_recall_curve(y_true, score)
        axis.plot(
            recall,
            precision,
            linewidth=1.4,
            color=MODEL_COLORS[name],
            label=f"{MODEL_LABELS[name]} (AP {average_precision_score(y_true, score):.3f})",
        )
    axis.set_xlabel("召回率 Recall")
    axis.set_ylabel("精确率 Precision")
    axis.legend(fontsize=7, loc="upper right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(pr_path, dpi=180)
    plt.close(figure)


def plot_confusion(y_true, predictions: dict, path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(6.5, 4.0))
    axes = axes.ravel()
    for index, name in enumerate(MODEL_NAMES):
        matrix = confusion_matrix(y_true, predictions[name], labels=[0, 1])
        row_sums = matrix.sum(axis=1, keepdims=True)
        rates = matrix / np.maximum(row_sums, 1)
        axis = axes[index]
        axis.imshow(rates, cmap="Blues", vmin=0, vmax=1)
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    f"{rates[row, column]:.1%}\n({matrix[row, column]:,})",
                    ha="center",
                    va="center",
                    color="white" if rates[row, column] > 0.55 else "#222222",
                    fontsize=7,
                )
        axis.set_xticks([0, 1], labels=["负向", "正向"], fontsize=7)
        axis.set_yticks([0, 1], labels=["负向", "正向"], fontsize=7)
        axis.set_xlabel("预测", fontsize=7)
        axis.set_ylabel("真实", fontsize=7)
        axis.set_title(MODEL_LABELS[name], fontsize=7.5)
    axes[-1].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_metrics(summary: dict, path: Path) -> None:
    metrics = ["macro_f1", "pr_auc", "roc_auc"]
    labels = {"macro_f1": "宏 F1", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"}
    width = 0.24
    positions = np.arange(len(MODEL_NAMES))
    figure, axis = plt.subplots(figsize=(6.8, 4.18))
    for offset, metric in enumerate(metrics):
        means = [summary[name][metric]["mean"] for name in MODEL_NAMES]
        stds = [summary[name][metric]["std"] for name in MODEL_NAMES]
        axis.bar(
            positions + (offset - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=2,
            color=[MODEL_COLORS[name] for name in MODEL_NAMES],
            alpha=[0.45, 0.7, 1.0][offset],
            edgecolor="white",
            linewidth=0.5,
        )
    axis.set_xticks(positions, labels=[MODEL_LABELS[name] for name in MODEL_NAMES], rotation=15, fontsize=7)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("5 折均值")
    legend = [Patch(facecolor="#666666", alpha=a, label=labels[m]) for a, m in zip([0.45, 0.7, 1.0], metrics)]
    axis.legend(handles=legend, fontsize=7)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_importance(importance: pd.DataFrame, path: Path) -> None:
    top = importance.head(20).iloc[::-1]
    figure, axis = plt.subplots(figsize=(6.2, 4.82))
    axis.barh(top["feature"], top["mean"], xerr=top["std"], color="#4C72B0")
    axis.set_xlabel("置换后 ROC-AUC 下降")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_shap(contributions: np.ndarray, base_value: float, feature_names: list[str], samples: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.27))
    for index in range(3):
        values = contributions[index]
        order = np.argsort(np.abs(values))[-10:][::-1]
        axes[index].barh(
            [feature_names[i] for i in order][::-1],
            [values[i] for i in order][::-1],
            color=["#C44E52" if values[i] > 0 else "#4C72B0" for i in order][::-1],
        )
        axes[index].set_title(
            f"{index + 1} {samples[index]['group']}  f={samples[index]['score']:.2f}  y={samples[index]['label']}",
            fontsize=6.5,
            pad=3,
        )
        axes[index].tick_params(labelsize=6.5)
        axes[index].set_xlabel("SHAP 值（正类）", fontsize=6.5)
        axes[index].grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="游戏评论正向/负向分类")
    parser.add_argument("--limit", type=int, default=None, help="只取多少行调试")
    parser.add_argument("--skip-heavy", action="store_true", help="跳过置换重要性与 SHAP")
    args = parser.parse_args()

    setup_fonts()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    frame = load_frame(args.limit)
    rows = len(frame)
    labels = frame["label"].to_numpy(dtype=np.int8)
    folds = frame["fold"].to_numpy(dtype=np.int8)
    print(f"建模样本 {rows:,} 行，正类 {labels.mean():.4f}")

    text = build_text(frame)
    categories = build_categories(frame)
    numbers = frame[NUMERIC_COLS].astype("float32")

    print("文本 hashing 向量化")
    vectorizer = HashingVectorizer(
        n_features=N_HASH,
        alternate_sign=False,
        norm=None,
        dtype=np.float32,
        token_pattern=r"(?u)\b\w\w+\b",
        lowercase=True,
    )
    text_counts = vectorizer.transform(text)
    del text
    gc.collect()
    print(f"文本矩阵 {text_counts.shape[0]:,} × {text_counts.shape[1]:,}，非零 {text_counts.nnz:,}")

    oof_score = {name: np.full(rows, np.nan, dtype=np.float32) for name in MODEL_NAMES}
    oof_pred = {name: np.full(rows, -1, dtype=np.int8) for name in MODEL_NAMES}
    fold_rows: list[dict] = []
    interpretation = None
    rng = np.random.default_rng(SEED)

    for fold in range(N_SPLITS):
        fold_started = time.perf_counter()
        train_mask = folds != fold
        valid_mask = folds == fold
        train_index = np.flatnonzero(train_mask)
        valid_index = np.flatnonzero(valid_mask)
        y_train = labels[train_index]
        y_valid = labels[valid_index]
        print(f"第 {fold} 折：训练 {len(y_train):,}，验证 {len(y_valid):,}")

        imputer = SimpleImputer(strategy="median", add_indicator=True)
        number_train = imputer.fit_transform(numbers.iloc[train_index])
        number_valid = imputer.transform(numbers.iloc[valid_index])
        scaler = StandardScaler()
        number_train = scaler.fit_transform(number_train).astype(np.float32)
        number_valid = scaler.transform(number_valid).astype(np.float32)
        feature_names = list(imputer.get_feature_names_out(NUMERIC_COLS))

        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=50,
            sparse_output=True,
            dtype=np.float32,
        )
        category_train = encoder.fit_transform(categories.iloc[train_index])
        category_valid = encoder.transform(categories.iloc[valid_index])

        tfidf = TfidfTransformer(sublinear_tf=True)
        text_train = tfidf.fit_transform(text_counts[train_index])
        text_valid = tfidf.transform(text_counts[valid_index])

        linear_train = sp.hstack(
            [sp.csr_matrix(number_train), category_train, text_train], format="csr"
        )
        linear_valid = sp.hstack(
            [sp.csr_matrix(number_valid), category_valid, text_valid], format="csr"
        )

        for name in ["dummy", "logistic_sgd", "linear_svm"]:
            model_started = time.perf_counter()
            _, score, pred = model_and_score(name, linear_train, y_train, linear_valid)
            seconds = time.perf_counter() - model_started
            oof_score[name][valid_index] = score
            oof_pred[name][valid_index] = pred
            fold_rows.append(
                metric_row(name, fold, y_valid, score, pred, seconds, len(y_train), len(y_valid))
            )
            print(f"  {MODEL_LABELS[name]}：{seconds:.1f} 秒")

        del linear_train, linear_valid, category_train, category_valid
        del number_train, number_valid
        gc.collect()

        svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED)
        sample_size = min(SVD_FIT_ROWS, text_train.shape[0])
        sample_index = rng.choice(text_train.shape[0], size=sample_size, replace=False)
        svd.fit(text_train[sample_index])
        svd_train = svd.transform(text_train).astype(np.float32)
        svd_valid = svd.transform(text_valid).astype(np.float32)

        tree_train = np.hstack(
            [imputer.transform(numbers.iloc[train_index]).astype(np.float32), svd_train]
        )
        tree_valid = np.hstack(
            [imputer.transform(numbers.iloc[valid_index]).astype(np.float32), svd_valid]
        )
        tree_names = feature_names + [f"svd_{index:02d}" for index in range(SVD_COMPONENTS)]

        for name in ["hist_gb", "random_forest"]:
            model_started = time.perf_counter()
            model, score, pred = model_and_score(name, tree_train, y_train, tree_valid)
            seconds = time.perf_counter() - model_started
            oof_score[name][valid_index] = score
            oof_pred[name][valid_index] = pred
            fold_rows.append(
                metric_row(name, fold, y_valid, score, pred, seconds, len(y_train), len(y_valid))
            )
            print(f"  {MODEL_LABELS[name]}：{seconds:.1f} 秒")
            if fold == 0 and name == "hist_gb":
                local_rows = min(PERMUTATION_ROWS, tree_valid.shape[0])
                interpretation = {
                    "model": model,
                    "features": tree_names,
                    "x": tree_valid[:local_rows].copy(),
                    "y": y_valid[:local_rows].copy(),
                    "index": valid_index[:local_rows].copy(),
                }

        del tree_train, tree_valid, svd_train, svd_valid
        del text_train, text_valid
        gc.collect()
        print(f"  第 {fold} 折用时 {time.perf_counter() - fold_started:.1f} 秒")

    metrics = pd.DataFrame(fold_rows)
    metrics.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")

    summary = {}
    for name in MODEL_NAMES:
        part = metrics.loc[metrics["model"] == name]
        summary[name] = {
            metric: {
                "mean": round(float(part[metric].mean()), 4),
                "std": round(float(part[metric].std(ddof=1)), 4),
            }
            for metric in METRIC_NAMES
        }
        summary[name]["seconds"] = round(float(part["seconds"].mean()), 1)

    best_name = max(MODEL_NAMES, key=lambda item: summary[item]["macro_f1"]["mean"])
    best_scores = metrics.loc[metrics["model"] == best_name, "macro_f1"].to_numpy()
    dummy_scores = metrics.loc[metrics["model"] == "dummy", "macro_f1"].to_numpy()
    if not np.allclose(best_scores, dummy_scores):
        stat, p_value = wilcoxon(best_scores, dummy_scores)
        test = {"test": "wilcoxon", "statistic": float(stat), "p_value": float(p_value)}
    else:
        test = {"test": "wilcoxon", "statistic": None, "p_value": None}

    plot_curves(
        labels,
        oof_score,
        FIGURES / "fig30_classify_roc.png",
        FIGURES / "fig31_classify_pr.png",
    )
    plot_confusion(labels, oof_pred, FIGURES / "fig32_classify_confusion.png")
    plot_metrics(summary, FIGURES / "fig35_classify_metrics.png")

    best_score = oof_score[best_name]
    error_rows = []
    for error_type, mask, order in [
        ("假正例", labels == 0, np.argsort(-best_score)),
        ("假负例", labels == 1, np.argsort(best_score)),
    ]:
        selected = []
        seen_asins = set()
        for index in order:
            if not mask[index]:
                continue
            asin = str(frame.iloc[index]["asin"])
            if asin in seen_asins:
                continue
            selected.append(index)
            seen_asins.add(asin)
            if len(selected) == 5:
                break
        for index in selected:
            row = frame.iloc[index]
            error_rows.append(
                {
                    "error_type": error_type,
                    "model": best_name,
                    "model_label": MODEL_LABELS[best_name],
                    "score": round(float(best_score[index]), 4),
                    "label": int(labels[index]),
                    "fold": int(folds[index]),
                    "reviewerID": row["reviewerID"],
                    "asin": row["asin"],
                    "verified": bool(row["verified"]),
                    "review_len": int(row["review_len"]),
                    "main_cat_clean": row["main_cat_clean"],
                    "category_leaf": row["category_leaf"],
                    "summary": str(row["summary"])[:200],
                    "reviewText": str(row["reviewText"])[:500].replace("\n", " "),
                }
            )
    pd.DataFrame(error_rows).to_csv(ERROR_CSV, index=False, encoding="utf-8-sig")

    importance = None
    shap_payload = None
    if not args.skip_heavy and interpretation is not None:
        print("置换重要性")
        permutation = permutation_importance(
            interpretation["model"],
            interpretation["x"],
            interpretation["y"],
            scoring="roc_auc",
            n_repeats=PERMUTATION_REPEATS,
            random_state=SEED,
            n_jobs=-1,
        )
        importance = pd.DataFrame(
            {
                "feature": interpretation["features"],
                "mean": permutation.importances_mean,
                "std": permutation.importances_std,
            }
        ).sort_values("mean", ascending=False)
        importance.to_csv(IMPORTANCE_CSV, index=False, encoding="utf-8-sig")
        plot_importance(importance, FIGURES / "fig33_classify_importance.png")

        print("SHAP 局部解释")
        import shap

        explainer = shap.TreeExplainer(interpretation["model"])
        hist_model = interpretation["model"]
        local_x = interpretation["x"]
        local_y = interpretation["y"]
        local_prob = hist_model.predict_proba(local_x)[:, 1]
        local_pred = (local_prob >= 0.5).astype(np.int8)
        pools = [
            ("假正例", np.flatnonzero((local_y == 0) & (local_pred == 1))),
            ("假负例", np.flatnonzero((local_y == 1) & (local_pred == 0))),
            ("真正例", np.flatnonzero((local_y == 1) & (local_pred == 1))),
        ]
        chosen: list[int] = []
        chosen_groups: list[str] = []
        for group_name, pool in pools:
            if len(pool) and int(pool[0]) not in chosen:
                chosen.append(int(pool[0]))
                chosen_groups.append(group_name)
        for index in range(len(local_x)):
            if len(chosen) >= 3:
                break
            if index not in chosen:
                chosen.append(index)
                chosen_groups.append("其他")
        sample_x = local_x[chosen]
        sample_index = interpretation["index"][chosen]
        shap_values = explainer.shap_values(sample_x)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_values = np.asarray(shap_values)
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(np.asarray(base_value).ravel()[-1])
        logits = hist_model.decision_function(sample_x)
        samples = []
        for offset, index in enumerate(sample_index):
            samples.append(
                {
                    "reviewerID": str(frame.iloc[index]["reviewerID"]),
                    "asin": str(frame.iloc[index]["asin"]),
                    "label": int(labels[index]),
                    "score": round(float(logits[offset]), 4),
                    "group": chosen_groups[offset],
                }
            )
        shap_payload = {
            "feature_names": interpretation["features"],
            "base_value": base_value,
            "samples": samples,
            "contributions": shap_values.tolist(),
        }
        SHAP_JSON.write_text(json.dumps(shap_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        plot_shap(shap_values, base_value, interpretation["features"], samples, FIGURES / "fig34_classify_shap.png")

    elapsed = time.perf_counter() - started
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rows": int(rows),
        "target": f"overall >= {POSITIVE_RATING_THRESHOLD}",
        "positive_rate": round(float(labels.mean()), 4),
        "n_splits": N_SPLITS,
        "n_model_features": len(MODEL_FEATURES),
        "numeric_features": NUMERIC_COLS,
        "categorical_features": CAT_COLS,
        "text": {
            "vectorizer": "HashingVectorizer + TfidfTransformer",
            "n_features": N_HASH,
            "max_chars": MAX_TEXT_CHARS,
            "svd_components": SVD_COMPONENTS,
            "svd_fit_rows": SVD_FIT_ROWS,
        },
        "models": summary,
        "best_model": best_name,
        "best_model_label": MODEL_LABELS[best_name],
        "wilcoxon_best_vs_dummy": test,
        "seconds": round(elapsed, 1),
        "hardware": hardware(),
        "outputs": {
            "metrics": str(METRICS_CSV),
            "errors": str(ERROR_CSV),
            "importance": str(IMPORTANCE_CSV) if importance is not None else None,
            "shap": str(SHAP_JSON) if shap_payload is not None else None,
        },
    }
    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    assert metrics["fold"].nunique() == N_SPLITS, "折数不完整"
    assert set(metrics["model"]) == set(MODEL_NAMES), "模型结果不完整"
    assert np.isfinite(best_score).all(), "存在未填的 OOF 分数"
    assert 0.0 <= metrics["roc_auc"].min() and metrics["roc_auc"].max() <= 1.0
    assert metrics["accuracy"].between(0.0, 1.0).all()
    assert summary[best_name]["macro_f1"]["mean"] >= summary["dummy"]["macro_f1"]["mean"]

    print(f"最佳模型：{MODEL_LABELS[best_name]}，宏 F1 {summary[best_name]['macro_f1']['mean']:.4f}")
    print(f"总用时 {elapsed:.1f} 秒，结果 {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
