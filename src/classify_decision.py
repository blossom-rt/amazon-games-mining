"""分类概率的决策曲线与业务阈值。

用 20 万条分层抽样、按 asin 分组的 5 折，训练文本 TF-IDF + 逻辑回归，
得到 OOF 概率；再用交叉拟合的 isotonic 校准，比较不同阈值下的净收益：

    net benefit = TP/n - FP/n * pt/(1-pt)

同时给出 F1 最优阈值、代价敏感阈值（FN=5×FP）和默认 0.5 的混淆矩阵。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

try:
    from .config import FEATURES_PARQUET, FIGURES, FONT_CANDIDATES, SEED, TABLES
except ImportError:
    from config import FEATURES_PARQUET, FIGURES, FONT_CANDIDATES, SEED, TABLES

OUT_JSON = TABLES / "classify_decision.json"
OUT_CURVE = TABLES / "classify_decision_curve.csv"
OUT_OOF = TABLES / "classify_decision_oof.parquet"
OUT_FIG = FIGURES / "fig48_decision_curve.png"
N_HASH = 2**18
N_SPLITS = 5


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probability >= left) & (probability < right if right < 1 else probability <= right)
        if mask.sum() == 0:
            continue
        total += mask.mean() * abs(y_true[mask].mean() - probability[mask].mean())
    return float(total)


def net_benefit(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> float:
    predicted = probability >= threshold
    tp = int(np.sum(predicted & (y_true == 1)))
    fp = int(np.sum(predicted & (y_true == 0)))
    n = len(y_true)
    return tp / n - fp / n * (threshold / (1.0 - threshold))


def threshold_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    predicted = probability >= threshold
    tp = int(np.sum(predicted & (y_true == 1)))
    fp = int(np.sum(predicted & (y_true == 0)))
    tn = int(np.sum(~predicted & (y_true == 0)))
    fn = int(np.sum(~predicted & (y_true == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "net_benefit": float(net_benefit(y_true, probability, threshold)),
    }


def main() -> None:
    started = time.perf_counter()
    setup_fonts()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    columns = [
        "reviewerID", "asin", "reviewText", "summary", "overall", "label",
        "in_scope", "fold", "verified", "main_cat_clean", "review_year",
    ]
    frame = pq.read_table(FEATURES_PARQUET, columns=columns).to_pandas()
    frame = frame.loc[frame["in_scope"]].reset_index(drop=True)
    if len(frame) > 200_000:
        positions = rng.choice(len(frame), size=200_000, replace=False)
        frame = frame.iloc[np.sort(positions)].reset_index(drop=True)

    y = frame["label"].to_numpy(dtype=np.int8)
    groups = frame["asin"].astype(str).to_numpy()
    text = (
        frame["reviewText"].fillna("").astype(str)
        + " "
        + frame["summary"].fillna("").astype(str)
    ).str.slice(0, 3000).tolist()

    vectorizer = HashingVectorizer(
        n_features=N_HASH,
        alternate_sign=False,
        norm=None,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    counts = vectorizer.transform(text)
    print(f"样本 {len(y):,}，正类 {y.mean():.4f}，哈希矩阵 {counts.shape}")

    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups), start=1):
        tfidf = TfidfTransformer(sublinear_tf=True)
        train_x = tfidf.fit_transform(counts[train_idx])
        valid_x = tfidf.transform(counts[valid_idx])
        model = SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=8,
            tol=None,
            class_weight="balanced",
            random_state=SEED + fold,
        )
        model.fit(train_x, y[train_idx])
        oof[valid_idx] = model.predict_proba(valid_x)[:, 1]
        print(f"折 {fold} 完成")

    raw_brier = brier_score_loss(y, oof)
    raw_log_loss = log_loss(y, oof)
    raw_ece = expected_calibration_error(y, oof)
    raw_auc = roc_auc_score(y, oof)

    calibrated = np.zeros_like(oof)
    for train_idx, valid_idx in splitter.split(np.zeros(len(y)), y, groups):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof[train_idx], y[train_idx])
        calibrated[valid_idx] = iso.predict(oof[valid_idx])
    calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)
    cal_brier = brier_score_loss(y, calibrated)
    cal_log_loss = log_loss(y, calibrated)
    cal_ece = expected_calibration_error(y, calibrated)
    cal_auc = roc_auc_score(y, calibrated)

    oof_frame = frame[
        ["reviewerID", "asin", "overall", "verified", "main_cat_clean", "review_year", "label"]
    ].copy()
    oof_frame["probability_raw"] = oof
    oof_frame["probability_calibrated"] = calibrated
    oof_frame.to_parquet(OUT_OOF, index=False)

    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    curve = pd.DataFrame(
        {
            "threshold": thresholds,
            "net_benefit_raw": [net_benefit(y, oof, value) for value in thresholds],
            "net_benefit_calibrated": [net_benefit(y, calibrated, value) for value in thresholds],
        }
    )
    prevalence = float(y.mean())
    curve["net_benefit_treat_all"] = prevalence - (1 - prevalence) * curve["threshold"] / (1 - curve["threshold"])
    curve.to_csv(OUT_CURVE, index=False, encoding="utf-8-sig")

    best_row = curve.loc[curve["net_benefit_calibrated"].idxmax()]
    best_threshold = float(best_row["threshold"])
    f1_candidates = [threshold_metrics(y, calibrated, value) for value in thresholds]
    f1_best = max(f1_candidates, key=lambda row: row["f1"])
    cost_candidates = [
        threshold_metrics(y, calibrated, value)
        for value in thresholds
    ]
    cost_best = min(cost_candidates, key=lambda row: 5 * row["fn"] + row["fp"])

    result = {
        "n_rows": int(len(y)),
        "positive_rate": prevalence,
        "n_folds": N_SPLITS,
        "seed": SEED,
        "raw": {
            "brier": float(raw_brier),
            "log_loss": float(raw_log_loss),
            "ece": float(raw_ece),
            "roc_auc": float(raw_auc),
        },
        "calibrated": {
            "brier": float(cal_brier),
            "log_loss": float(cal_log_loss),
            "ece": float(cal_ece),
            "roc_auc": float(cal_auc),
        },
        "best_net_benefit_threshold": best_threshold,
        "best_net_benefit": float(best_row["net_benefit_calibrated"]),
        "f1_best": f1_best,
        "cost_sensitive_fn5": cost_best,
        "threshold_0_5": threshold_metrics(y, calibrated, 0.5),
        "notes": [
            "20 万条分层抽样，按 asin 分组 5 折；概率为 OOF。",
            "isotonic 用交叉拟合：每折的校准器只在其余折 OOF 上拟合。",
            "净收益按每 1 条样本计；代价敏感阈值最小化 5*FN+FP。",
        ],
        "outputs": {"curve": str(OUT_CURVE), "oof": str(OUT_OOF), "figure": str(OUT_FIG)},
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(curve["threshold"], curve["net_benefit_raw"], label="原始概率", color="#4c72b0")
    axis.plot(curve["threshold"], curve["net_benefit_calibrated"], label="isotonic 校准", color="#55a868")
    axis.plot(curve["threshold"], curve["net_benefit_treat_all"], "--", label="全部判正", color="#8c8c8c")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(best_threshold, color="#c44e52", linestyle=":", label=f"净收益最优 {best_threshold:.2f}")
    axis.axvline(cost_best["threshold"], color="#dd8452", linestyle="-.", label=f"FN=5×FP {cost_best['threshold']:.2f}")
    axis.set_xlabel("判定阈值")
    axis.set_ylabel("净收益")
    axis.set_ylim(-0.02, 0.82)
    axis.set_title("分类决策曲线（20 万抽样，按 asin 5 折）")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(
        f"raw Brier {raw_brier:.4f} / ECE {raw_ece:.4f}；校准后 Brier {cal_brier:.4f} / ECE {cal_ece:.4f}"
    )
    print(f"净收益最优阈值 {best_threshold:.2f}；F1 最优 {f1_best['threshold']:.2f}；代价阈值 {cost_best['threshold']:.2f}")
    print(f"写入 {OUT_JSON}；耗时 {time.perf_counter() - started:.1f} 秒")


if __name__ == "__main__":
    main()
