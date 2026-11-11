"""P2 融合与特征：构建评论 × 商品宽表，并固定分层分组折。

一行一条评论（2,565,349 行），把商品属性左连接到评论上，再生成时间、
文本、价格与交互特征。折划分用 StratifiedGroupKFold，分组键为 asin，
保证同一商品的评论只落在一个折里；折在特征工程之前生成，后续脚本只
读取 fold 列，不得重新划分。

用法：
    python src/build_features.py
    python src/build_features.py --force
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold

try:
    from .config import (
        DATA_PROCESSED,
        FEATURES_PARQUET,
        FOLDS_PARQUET,
        META_FILE,
        N_SPLITS,
        POSITIVE_RATING_THRESHOLD,
        REVIEW_FILE,
        SEED,
        TABLES,
    )
    from .parse import _hardware
except ImportError:
    from config import (
        DATA_PROCESSED,
        FEATURES_PARQUET,
        FOLDS_PARQUET,
        META_FILE,
        N_SPLITS,
        POSITIVE_RATING_THRESHOLD,
        REVIEW_FILE,
        SEED,
        TABLES,
    )
    from parse import _hardware

REVIEWS_PARQUET = DATA_PROCESSED / "reviews.parquet"
META_PARQUET = DATA_PROCESSED / "meta.parquet"
LOG_OUT = TABLES / "feature_summary.json"

TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
LEADING_BY_RE = re.compile(r"^by\s+", flags=re.IGNORECASE)

# 参与建模的列，不含主键、目标、折号与原始文本
MODEL_FEATURES = [
    "verified",
    "review_year",
    "review_month",
    "review_dow",
    "review_len",
    "review_word_count",
    "review_sentence_count",
    "summary_len",
    "has_style",
    "has_vote",
    "vote_count",
    "has_review_image",
    "image_count",
    "has_meta",
    "title_len",
    "price",
    "price_missing",
    "log_price",
    "main_cat_clean",
    "category_leaf",
    "category_depth",
    "also_buy_len",
    "also_view_len",
    "feature_count",
    "description_len",
    "rank_count",
    "image_count_meta",
    "tech1_present",
    "tech2_present",
    "days_since_product",
    "verified_x_log_len",
    "log_price_x_verified",
    "helpful_x_len",
    "days_since_verified",
]

# 在原始字段之外新造的列，报告里单独说明
DERIVED_FEATURES = [
    "review_year",
    "review_month",
    "review_dow",
    "review_len",
    "review_word_count",
    "review_sentence_count",
    "summary_len",
    "has_style",
    "has_vote",
    "vote_count",
    "has_review_image",
    "has_meta",
    "title_len",
    "price_missing",
    "log_price",
    "category_depth",
    "also_buy_len",
    "also_view_len",
    "feature_count",
    "description_len",
    "rank_count",
    "image_count_meta",
    "tech1_present",
    "tech2_present",
    "days_since_product",
    "verified_x_log_len",
    "log_price_x_verified",
    "helpful_x_len",
    "days_since_verified",
]

FORBIDDEN_IN_FEATURES = {"overall", "label", "fold", "reviewText", "summary"}

OUTPUT_COLUMNS = [
    "reviewerID",
    "asin",
    "unixReviewTime",
    "overall",
    "label",
    "fold",
    "verified",
    "review_year",
    "review_month",
    "review_dow",
    "review_len",
    "review_word_count",
    "review_sentence_count",
    "summary_len",
    "has_style",
    "has_vote",
    "vote_count",
    "has_review_image",
    "image_count",
    "reviewText",
    "summary",
    "has_meta",
    "title_clean",
    "title_len",
    "brand_clean",
    "price",
    "price_missing",
    "log_price",
    "main_cat_clean",
    "category_leaf",
    "category_depth",
    "also_buy_len",
    "also_view_len",
    "feature_count",
    "description_len",
    "rank_count",
    "image_count_meta",
    "tech1_present",
    "tech2_present",
    "days_since_product",
    "verified_x_log_len",
    "log_price_x_verified",
    "helpful_x_len",
    "days_since_verified",
]


def clean_text(value) -> str | None:
    if not isinstance(value, str):
        return None
    text = html.unescape(value)
    text = TAG_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text or None


def clean_brand(value) -> str | None:
    if not isinstance(value, str):
        return None
    text = SPACE_RE.sub(" ", value).strip()
    text = LEADING_BY_RE.sub("", text).strip()
    return text or None


def clean_main_cat(value) -> str | None:
    text = clean_text(value)
    if text is None or text.lower() == "none":
        return None
    return text


def category_leaf(value) -> str | None:
    if value is None:
        return None
    try:
        items = list(value)
    except TypeError:
        return None
    for item in reversed(items):
        text = clean_text(str(item))
        if text:
            return text
    return None


def category_depth(value) -> int:
    if value is None:
        return 0
    try:
        items = list(value)
    except TypeError:
        return 0
    return sum(1 for item in items if clean_text(str(item)))


def make_folds(asin: pd.Series, label: pd.Series) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold = np.full(len(label), -1, dtype=np.int8)
    for index, (_, valid) in enumerate(
        splitter.split(np.zeros(len(label), dtype=np.int8), label, groups=asin)
    ):
        fold[valid] = index
    assert (fold >= 0).all(), "存在未分配折的样本"
    return fold


def build_product_frame(meta: pd.DataFrame) -> pd.DataFrame:
    prod = pd.DataFrame(
        {
            "asin": meta["asin"],
            "title_clean": meta["title"].map(clean_text),
            "brand_clean": meta["brand"].map(clean_brand),
            "price": meta["price"],
            "main_cat_clean": meta["main_cat"].map(clean_main_cat),
            "category_leaf": meta["category"].map(category_leaf),
            "category_depth": meta["category"].map(category_depth),
            "also_buy_len": meta["also_buy"].map(len),
            "also_view_len": meta["also_view"].map(len),
            "feature_count": meta["feature"].map(len),
            "description_len": meta["description"].str.len(),
            "rank_count": meta["rank"].map(len),
            "image_count_meta": meta["imageURL"].map(len),
            "product_date": pd.to_datetime(meta["date"], errors="coerce"),
            "tech1_present": meta["tech1"].notna(),
            "tech2_present": meta["tech2"].notna(),
        }
    )
    prod["title_len"] = prod["title_clean"].str.len().fillna(0).astype("int32")
    prod["price_missing"] = prod["price"].isna()
    prod["log_price"] = np.log1p(prod["price"])
    prod["product_days"] = (
        prod["product_date"] - pd.Timestamp("1970-01-01")
    ).dt.days
    return prod.drop(columns=["product_date"])


def add_review_features(reviews: pd.DataFrame) -> pd.DataFrame:
    text = reviews["reviewText"].fillna("")
    summary = reviews["summary"].fillna("")
    has_text = text.str.len() > 0

    reviews["review_len"] = text.str.len().astype("int32")
    words = (text.str.count(r"\s+") + 1).astype("int32")
    reviews["review_word_count"] = words.where(has_text, 0).astype("int32")
    sentences = (text.str.count(r"[.!?]+") + 1).astype("int32")
    reviews["review_sentence_count"] = sentences.where(has_text, 0).astype("int32")
    reviews["summary_len"] = summary.str.len().astype("int32")

    reviews["has_style"] = reviews["style"].notna()
    reviews["has_vote"] = reviews["vote"].notna()
    reviews["vote_count"] = reviews["vote"].fillna(0).astype("int32")
    reviews["image_count"] = reviews["image_count"].fillna(0).astype("int32")
    reviews["has_review_image"] = reviews["image_count"] > 0

    timestamp = pd.to_datetime(reviews["unixReviewTime"], unit="s")
    reviews["review_year"] = timestamp.dt.year.astype("int16")
    reviews["review_month"] = timestamp.dt.month.astype("int8")
    reviews["review_dow"] = timestamp.dt.day_of_week.astype("int8")
    return reviews


def build_features(force: bool) -> dict:
    if FEATURES_PARQUET.exists() and not force:
        raise SystemExit(f"输出已存在，加 --force 覆盖: {FEATURES_PARQUET}")
    TABLES.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    reviews = pq.read_table(REVIEWS_PARQUET).to_pandas()
    meta = pq.read_table(META_PARQUET).to_pandas()
    rows = len(reviews)
    print(f"评论 {rows:,} 行，商品 {len(meta):,} 行")

    assert reviews["overall"].notna().all(), "存在缺失评分，无法确定标签"
    assert reviews["unixReviewTime"].notna().all(), "存在缺失时间戳"
    label = (reviews["overall"] >= POSITIVE_RATING_THRESHOLD).astype("int8")

    # 折在任何目标相关变换、编码、填补之前固定下来
    fold = make_folds(reviews["asin"], label)
    reviews["label"] = label
    reviews["fold"] = fold

    reviews = add_review_features(reviews)

    prod = build_product_frame(meta)
    assert prod["asin"].is_unique, "商品表 asin 不唯一"
    has_meta = reviews["asin"].isin(set(prod["asin"]))

    features = reviews.merge(prod, on="asin", how="left", validate="many_to_one")
    features["has_meta"] = has_meta.to_numpy()
    features["price_missing"] = features["price_missing"].fillna(True).astype(bool)
    features["tech1_present"] = features["tech1_present"].fillna(False).astype(bool)
    features["tech2_present"] = features["tech2_present"].fillna(False).astype(bool)

    review_days = (features["unixReviewTime"] // 86400).astype("int64")
    days = review_days - features["product_days"]
    negative_days = int((days < 0).sum())
    features["days_since_product"] = days.clip(lower=0)

    verified = features["verified"].astype("int8")
    log_len = np.log1p(features["review_len"])
    features["verified_x_log_len"] = verified * log_len
    features["log_price_x_verified"] = features["log_price"] * verified
    features["helpful_x_len"] = np.log1p(features["vote_count"]) * log_len
    features["days_since_verified"] = features["days_since_product"] * verified

    features["price"] = features["price"].astype("float32")
    features["log_price"] = features["log_price"].astype("float32")
    features["days_since_product"] = features["days_since_product"].astype("float32")
    features["verified_x_log_len"] = features["verified_x_log_len"].astype("float32")
    features["log_price_x_verified"] = features["log_price_x_verified"].astype("float32")
    features["helpful_x_len"] = features["helpful_x_len"].astype("float32")
    features["days_since_verified"] = features["days_since_verified"].astype("float32")

    features = features[OUTPUT_COLUMNS]
    features.to_parquet(FEATURES_PARQUET, index=False, compression="zstd")

    folds = features[["reviewerID", "asin", "label", "fold"]].copy()
    folds.to_parquet(FOLDS_PARQUET, index=False, compression="zstd")

    fold_sizes = features["fold"].value_counts().sort_index()
    fold_rates = features.groupby("fold")["label"].mean().sort_index()
    group_folds = features[["asin", "fold"]].drop_duplicates()

    assert len(features) == rows, "宽表行数与评论表不一致"
    assert features["label"].isin([0, 1]).all(), "标签不是 0/1"
    assert features["fold"].between(0, N_SPLITS - 1).all(), "折号越界"
    assert len(MODEL_FEATURES) >= 20, "建模特征不足 20 列"
    assert len(DERIVED_FEATURES) >= 8, "衍生特征不足 8 个"
    assert not (set(MODEL_FEATURES) & FORBIDDEN_IN_FEATURES), "特征里混入了目标或文本"
    assert features["price"].dropna().ge(0).all(), "存在负价格"
    assert features["review_len"].ge(0).all(), "存在负文本长度"
    assert not group_folds["asin"].duplicated().any(), "同一 asin 跨了多个折"
    assert fold_sizes.min() > 0, "存在空折"
    assert fold_rates.between(0.72, 0.76).all(), f"各折正类率偏差过大: {fold_rates.round(4).to_dict()}"

    elapsed = time.perf_counter() - started
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rows": int(rows),
        "columns": int(features.shape[1]),
        "target": f"overall >= {POSITIVE_RATING_THRESHOLD}",
        "positive_rate": round(float(features["label"].mean()), 4),
        "n_groups": int(features["asin"].nunique()),
        "n_model_features": len(MODEL_FEATURES),
        "n_derived_features": len(DERIVED_FEATURES),
        "model_features": MODEL_FEATURES,
        "derived_features": DERIVED_FEATURES,
        "fold_sizes": {int(k): int(v) for k, v in fold_sizes.items()},
        "fold_positive_rates": {int(k): round(float(v), 4) for k, v in fold_rates.items()},
        "has_meta_rows": int(features["has_meta"].sum()),
        "without_meta_rows": int((~features["has_meta"]).sum()),
        "negative_days_clipped": negative_days,
        "missing_rates": {
            column: round(float(features[column].isna().mean()), 4)
            for column in [
                "price",
                "brand_clean",
                "main_cat_clean",
                "category_leaf",
                "days_since_product",
                "summary",
                "reviewText",
            ]
        },
        "seconds": round(elapsed, 1),
        "hardware": _hardware(),
        "outputs": {
            "features": {
                "path": str(FEATURES_PARQUET),
                "size_mb": round(FEATURES_PARQUET.stat().st_size / 1024**2, 1),
            },
            "folds": {
                "path": str(FOLDS_PARQUET),
                "size_mb": round(FOLDS_PARQUET.stat().st_size / 1024**2, 1),
            },
        },
    }
    LOG_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"宽表 {features.shape[0]:,} 行 × {features.shape[1]} 列")
    print(f"建模特征 {len(MODEL_FEATURES)} 个，衍生特征 {len(DERIVED_FEATURES)} 个")
    print("各折行数", summary["fold_sizes"])
    print("各折正类率", summary["fold_positive_rates"])
    print(f"写入 {FEATURES_PARQUET}，{summary['outputs']['features']['size_mb']} MB")
    print(f"用时 {summary['seconds']} 秒，日志 {LOG_OUT}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="构建评论 × 商品宽表与固定折")
    parser.add_argument("--force", action="store_true", help="覆盖已有输出")
    args = parser.parse_args()
    build_features(force=args.force)


if __name__ == "__main__":
    main()
