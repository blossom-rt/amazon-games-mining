"""P1.5 数据治理验证：一致性检查、缺失填补对比与异常值检测。

这个脚本不修改 features.parquet，只读取现有产物并输出一份可复核的报告：

- 一致性：规模、主键与重复、时区、单位、连接基数、类别编码与脏数据；
- 缺失填补：在观测价格上人为遮蔽 20%，比较 3–4 种策略的回填误差；
- 异常值：IQR、z-score 与 IsolationForest 三种方法的检出量与重叠。

用法：
    python src/governance.py
    python src/governance.py --force
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .build_features import build_product_frame
    from .config import DATA_PROCESSED, DOCS, FEATURES_PARQUET, SEED, TABLES
    from .parse import _hardware
except ImportError:
    from build_features import build_product_frame
    from config import DATA_PROCESSED, DOCS, FEATURES_PARQUET, SEED, TABLES
    from parse import _hardware

REVIEWS_PARQUET = DATA_PROCESSED / "reviews.parquet"
META_PARQUET = DATA_PROCESSED / "meta.parquet"
REPORT_JSON = TABLES / "governance_report.json"
REPORT_MD = DOCS / "governance_report.md"

FEATURE_COLUMNS = [
    "reviewerID",
    "asin",
    "unixReviewTime",
    "overall",
    "label",
    "fold",
    "in_scope",
    "review_len",
    "review_word_count",
    "review_sentence_count",
    "summary_len",
    "vote_count",
    "image_count",
    "verified",
    "review_year",
    "review_month",
    "review_dow",
    "has_meta",
    "price",
    "price_missing",
    "main_cat_clean",
    "category_leaf",
    "category_depth",
    "also_buy_len",
    "also_view_len",
    "feature_count",
    "description_len",
    "rank_count",
    "image_count_meta",
    "title_len",
    "days_since_product",
]

META_COLUMNS = [
    "asin",
    "title",
    "brand",
    "price_raw",
    "price",
    "main_cat",
    "category",
    "also_buy",
    "also_view",
    "feature",
    "description",
    "rank",
    "imageURL",
    "date",
    "tech1",
    "tech2",
]

IMPUTE_NUMERIC = [
    "also_buy_len",
    "also_view_len",
    "feature_count",
    "description_len",
    "rank_count",
    "image_count_meta",
    "category_depth",
    "title_len",
]


def _load_features() -> pd.DataFrame:
    if not FEATURES_PARQUET.exists():
        raise SystemExit(f"缺少宽表，请先运行 python src/build_features.py: {FEATURES_PARQUET}")
    table = pq.read_table(FEATURES_PARQUET, columns=FEATURE_COLUMNS)
    return table.to_pandas()


def _load_meta() -> pd.DataFrame:
    if not META_PARQUET.exists():
        raise SystemExit(f"缺少商品表，请先运行 python src/parse.py: {META_PARQUET}")
    table = pq.read_table(META_PARQUET, columns=META_COLUMNS)
    return table.to_pandas()


def _cat_items(value) -> list:
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []


def _cat_has(value, needle: str) -> bool:
    return any(needle in str(item) for item in _cat_items(value))


def _consistency(features: pd.DataFrame, meta: pd.DataFrame) -> dict:
    expected_rows = int(pq.ParquetFile(REVIEWS_PARQUET).metadata.num_rows)
    assert len(features) == expected_rows, "宽表行数与评论 parquet 不一致"

    key_cols = ["reviewerID", "asin", "unixReviewTime"]
    duplicate_rows = int(features.duplicated(subset=key_cols).sum())
    group_sizes = features.groupby(key_cols, sort=False).size()
    duplicate_groups = int((group_sizes > 1).sum())
    max_group = int(group_sizes.max())

    in_scope = features["in_scope"]
    model = features.loc[in_scope]
    fold_per_asin = model.groupby("asin")["fold"].nunique()
    asin_cross_fold = int((fold_per_asin > 1).sum())

    utc_time = pd.to_datetime(features["unixReviewTime"], unit="s", utc=True)
    year_mismatch = int((utc_time.dt.year.astype("int32") != features["review_year"].astype("int32")).sum())

    price_all = features["price"].dropna()
    product_price = features.drop_duplicates("asin")["price"]
    price_positive = product_price[product_price > 0]

    main_cat_raw = meta["main_cat"].fillna("").astype(str)
    price_raw = meta["price_raw"].fillna("").astype(str).str.strip()
    category_empty = int(sum(1 for value in meta["category"] if len(_cat_items(value)) == 0))

    checks: list[dict] = []

    def check(name: str, condition: bool) -> None:
        checks.append({"name": name, "passed": bool(condition)})
        assert condition, f"一致性断言失败: {name}"

    check("商品 asin 唯一", bool(meta["asin"].is_unique))
    check("宽表行数等于评论 parquet", len(features) == expected_rows)
    check("标签只含 0/1", bool(features["label"].isin([0, 1]).all()))
    check("游戏本体折号在 0–4", bool(model["fold"].between(0, 4).all()))
    check("范围外折号为 -1", bool((features.loc[~in_scope, "fold"] == -1).all()))
    check("游戏本体评论不少于 100 万", len(model) >= 1_000_000)
    check("同一 asin 不跨折", asin_cross_fold == 0)
    check("价格非负", bool(price_all.ge(0).all()))
    check("派生年份与 UTC 一致", year_mismatch == 0)
    check(
        "关键数值列类型正确",
        all(
            pd.api.types.is_numeric_dtype(features[column])
            for column in ["overall", "label", "fold", "price", "review_len", "unixReviewTime"]
        ),
    )

    return {
        "reviews": {
            "rows": int(len(features)),
            "unique_reviewer": int(features["reviewerID"].nunique()),
            "unique_asin": int(features["asin"].nunique()),
            "label_counts": {
                "0": int((features["label"] == 0).sum()),
                "1": int((features["label"] == 1).sum()),
            },
            "positive_rate": round(float(features["label"].mean()), 4),
            "in_scope_rows": int(in_scope.sum()),
            "out_scope_rows": int((~in_scope).sum()),
            "in_scope_share": round(float(in_scope.mean()), 4),
            "in_scope_positive_rate": round(float(model["label"].mean()), 4),
            "verified_rate": round(float(features["verified"].mean()), 4),
            "unix_min": int(features["unixReviewTime"].min()),
            "unix_max": int(features["unixReviewTime"].max()),
            "date_min": utc_time.min().isoformat(),
            "date_max": utc_time.max().isoformat(),
            "year_min": int(features["review_year"].min()),
            "year_max": int(features["review_year"].max()),
            "has_meta": int(features["has_meta"].sum()),
            "without_meta": int((~features["has_meta"]).sum()),
            "days_since_null_share": round(float(features["days_since_product"].isna().mean()), 4),
            "duplicate_key_rows": duplicate_rows,
            "duplicate_key_groups": duplicate_groups,
            "max_duplicate_group": max_group,
            "asin_cross_fold": asin_cross_fold,
        },
        "meta": {
            "rows": int(len(meta)),
            "unique_asin": int(meta["asin"].nunique()),
            "price_notnull": int(meta["price"].notna().sum()),
            "price_observed_positive": int(len(price_positive)),
            "review_rows_with_price": int(price_all.notna().sum()),
            "price_zero": int((meta["price"] == 0).sum()),
            "price_negative": int((meta["price"] < 0).sum()),
            "price_min": round(float(meta["price"].min()), 2),
            "price_max": round(float(meta["price"].max()), 2),
            "brand_null": int(meta["brand"].isna().sum()),
            "main_cat_null": int(meta["main_cat"].isna().sum()),
            "main_cat_with_amp": int(main_cat_raw.str.contains("&amp;", regex=False).sum()),
            "main_cat_with_span": int(main_cat_raw.str.contains("</span>", regex=False).sum()),
            "main_cat_literal_none": int((main_cat_raw.str.strip().str.lower() == "none").sum()),
            "category_empty": category_empty,
            "category_with_span": int(meta["category"].map(lambda value: _cat_has(value, "</span>")).sum()),
            "category_with_amp": int(meta["category"].map(lambda value: _cat_has(value, "&amp;")).sum()),
            "price_raw_empty": int((price_raw == "").sum()),
            "price_raw_div": int(price_raw.str.startswith("<div", na=False).sum()),
            "price_raw_span": int(price_raw.str.startswith("<span", na=False).sum()),
            "price_raw_script": int(price_raw.str.startswith("<script", na=False).sum()),
            "also_buy_nonempty": int((meta["also_buy"].map(len) > 0).sum()),
            "also_buy_max": int(meta["also_buy"].map(len).max()),
        },
        "join": {
            "meta_rows": int(len(meta)),
            "unique_meta_asin": int(meta["asin"].nunique()),
            "feature_rows": int(len(features)),
            "review_to_meta": "N:1",
            "meta_to_review": "1:N",
            "coverage": round(float(features["has_meta"].mean()), 4),
        },
        "timezone": {
            "source": "unixReviewTime 为 UTC epoch 秒",
            "derived": "review_year/month/dow 均按 UTC 提取",
            "reviewTime_string_used": False,
            "year_mismatch": year_mismatch,
        },
        "units": {
            "price": "美元",
            "review_len": "字符数",
            "days_since_product": "天，负值截为 0",
        },
        "types": {
            "review": {
                column: str(features[column].dtype)
                for column in [
                    "reviewerID",
                    "asin",
                    "unixReviewTime",
                    "overall",
                    "label",
                    "fold",
                    "in_scope",
                    "price",
                    "review_len",
                ]
            },
            "meta": {
                column: str(meta[column].dtype)
                for column in ["asin", "price", "main_cat", "category", "date"]
            },
        },
        "assertions": checks,
    }


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "medae": round(float(median_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def _imputation(meta: pd.DataFrame) -> dict:
    product = build_product_frame(meta)
    product = product[product["price"].notna() & (product["price"] > 0)].copy()
    product["main_cat_clean"] = product["main_cat_clean"].fillna("未知")
    counts = product["main_cat_clean"].value_counts()
    product["stratum"] = product["main_cat_clean"].where(
        product["main_cat_clean"].map(counts) >= 10, "其他"
    )

    train_idx, test_idx = train_test_split(
        np.arange(len(product)),
        test_size=0.2,
        random_state=SEED,
        stratify=product["stratum"],
    )
    train = product.iloc[train_idx]
    test = product.iloc[test_idx]
    y_train = train["price"].astype("float64").to_numpy()
    y_test = test["price"].astype("float64").to_numpy()

    numeric = product[IMPUTE_NUMERIC].astype("float64").fillna(0.0)
    strategies: list[dict] = []

    global_median = float(np.median(y_train))
    pred = np.full(len(test), global_median, dtype="float64")
    strategies.append(
        {
            "name": "全局中位数",
            "detail": "训练集价格中位数，忽略品类",
            **_score(y_test, pred),
        }
    )

    category_median = train.groupby("main_cat_clean")["price"].median()
    pred = (
        test["main_cat_clean"].map(category_median).fillna(global_median).to_numpy(dtype="float64")
    )
    strategies.append(
        {
            "name": "品类中位数",
            "detail": "按 main_cat_clean 分组取中位数，缺失组回退全局中位数",
            **_score(y_test, pred),
        }
    )

    train_matrix = numeric.iloc[train_idx].copy()
    train_matrix["price"] = y_train
    test_matrix = numeric.iloc[test_idx].copy()
    test_matrix["price"] = np.nan

    knn = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
        ]
    )
    knn.fit(train_matrix)
    pred = knn.transform(test_matrix)[:, -1]
    strategies.append(
        {
            "name": "KNN 近邻",
            "detail": "标准化后取 5 个近邻加权平均，特征含商品列表长度与文本长度",
            **_score(y_test, pred),
        }
    )

    iterative_error = None
    try:
        iterative = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("imputer", IterativeImputer(max_iter=10, random_state=SEED)),
            ]
        )
        iterative.fit(train_matrix)
        pred = iterative.transform(test_matrix)[:, -1]
        strategies.append(
            {
                "name": "迭代回归",
                "detail": "IterativeImputer 以其余商品特征迭代回归价格",
                **_score(y_test, pred),
            }
        )
    except Exception as exc:  # 迭代填补在部分 sklearn 版本上不稳定
        iterative_error = f"{type(exc).__name__}: {exc}"

    best = min(strategies, key=lambda item: item["mae"])
    return {
        "target": "price",
        "n_observed_positive": int(len(product)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "masking": "在观测价格中按品类分层抽 20% 作为测试，其余作为训练",
        "strategies": strategies,
        "best": best["name"],
        "iterative_error": iterative_error,
        "limitations": [
            "测试集来自已观测价格，无法代表 83% 未观测价格的 MNAR 机制",
            "这里的误差是回填误差，不等于下游分类收益；P4 需在训练折内重新拟合",
        ],
    }


def _outliers(features: pd.DataFrame) -> dict:
    product_price = features.drop_duplicates("asin")["price"]
    price = product_price.dropna().to_numpy(dtype="float64")
    positive = price[price > 0]
    q1, q3 = np.quantile(positive, [0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    lengths = features["review_len"].dropna().to_numpy(dtype="float64")
    len_q1, len_q3 = np.quantile(lengths, [0.25, 0.75])
    len_iqr = len_q3 - len_q1
    len_lower = len_q1 - 1.5 * len_iqr
    len_upper = len_q3 + 1.5 * len_iqr
    len_mean = float(lengths.mean())
    len_std = float(lengths.std(ddof=0))
    z_flags = np.abs((lengths - len_mean) / len_std) > 3

    sample = features.sample(n=min(200_000, len(features)), random_state=SEED)
    outlier_features = [
        "review_len",
        "review_word_count",
        "review_sentence_count",
        "summary_len",
        "vote_count",
        "image_count",
    ]
    matrix = sample[outlier_features].astype("float64").fillna(0.0)
    matrix = StandardScaler().fit_transform(matrix)
    forest = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=SEED,
        n_jobs=-1,
    )
    forest_flags = forest.fit_predict(matrix) == -1
    sample_iqr_flags = (sample["review_len"] > len_upper) | (sample["review_len"] < len_lower)
    overlap = int((forest_flags & sample_iqr_flags.to_numpy()).sum())

    return {
        "price_iqr": {
            "scope": "商品价格，仅 price>0",
            "n": int(len(positive)),
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "lower": round(float(lower), 2),
            "upper": round(float(upper), 2),
            "below": int((positive < lower).sum()),
            "above": int((positive > upper).sum()),
            "zero_count": int((price == 0).sum()),
        },
        "review_len_iqr": {
            "scope": "评论正文字符数，全量",
            "n": int(len(lengths)),
            "q1": round(float(len_q1), 2),
            "q3": round(float(len_q3), 2),
            "lower": round(float(len_lower), 2),
            "upper": round(float(len_upper), 2),
            "below": int((lengths < len_lower).sum()),
            "above": int((lengths > len_upper).sum()),
        },
        "review_len_zscore": {
            "scope": "评论正文字符数，全量",
            "mean": round(len_mean, 2),
            "std": round(len_std, 2),
            "threshold": "|z| > 3",
            "flagged": int(z_flags.sum()),
            "share": round(float(z_flags.mean()), 4),
        },
        "isolation_forest": {
            "scope": "评论数值特征，20 万条随机样本",
            "n": int(len(sample)),
            "features": outlier_features,
            "contamination": 0.01,
            "flagged": int(forest_flags.sum()),
            "share": round(float(forest_flags.mean()), 4),
            "overlap_with_iqr": overlap,
        },
        "treatment": [
            "评论长度与词数进入模型前做 log1p，并在训练折内按 p99 截断；保留原始值供审计",
            "价格不删除 0 值，建模时用 price_missing 指示缺失，观测价格在训练折内按 p1/p99 缩尾",
            "IsolationForest 只用于抽样人工核查，不直接删行；长评论可能是真实长文本",
        ],
    }


def _markdown(report: dict) -> str:
    consistency = report["consistency"]
    reviews = consistency["reviews"]
    meta = consistency["meta"]
    imputation = report["imputation"]
    outliers = report["outliers"]

    lines: list[str] = []
    lines.append("# 数据治理验证报告")
    lines.append("")
    lines.append(
        "本报告对应表 1 中的数据治理要求：一致性检查、缺失填补策略对比与异常值检测。"
        "所有数字由 `python src/governance.py` 生成，原始结果见 "
        "`results/tables/governance_report.json`。"
    )
    lines.append("")
    lines.append("## 一致性检查")
    lines.append("")
    lines.append("### 规模与主键")
    lines.append("")
    lines.append("| 项目 | 数值 |")
    lines.append("|---|---|")
    lines.append(f"| 评论宽表行数 | {reviews['rows']:,} |")
    lines.append(f"| 游戏本体评论 | {reviews['in_scope_rows']:,}（{reviews['in_scope_share']:.2%}） |")
    lines.append(f"| 范围外评论 | {reviews['out_scope_rows']:,} |")
    lines.append(f"| 唯一评论人 | {reviews['unique_reviewer']:,} |")
    lines.append(f"| 唯一商品 asin | {reviews['unique_asin']:,} |")
    lines.append(f"| 商品表行数 | {meta['rows']:,} |")
    lines.append(f"| 商品表唯一 asin | {meta['unique_asin']:,} |")
    lines.append(f"| 评论正向占比 | {reviews['positive_rate']:.2%} |")
    lines.append(f"| 游戏本体正向占比 | {reviews['in_scope_positive_rate']:.2%} |")
    lines.append(f"| 已验证购买占比 | {reviews['verified_rate']:.2%} |")
    lines.append("")
    lines.append(
        f"评论表没有单一主键：`(reviewerID, asin, unixReviewTime)` 有 "
        f"{reviews['duplicate_key_rows']:,} 行重复、{reviews['duplicate_key_groups']:,} 组，"
        f"最大一组 {reviews['max_duplicate_group']} 行。商品表以 `asin` 为主键，"
        "P0 已去掉 12,908 条完全重复记录。"
    )
    lines.append("")
    lines.append("### 时间、单位与连接")
    lines.append("")
    lines.append("| 项目 | 结果 |")
    lines.append("|---|---|")
    lines.append(f"| 评论时间范围 | {reviews['date_min']} 至 {reviews['date_max']} |")
    lines.append(f"| 年份范围 | {reviews['year_min']}–{reviews['year_max']} |")
    lines.append(f"| 时区 | epoch 按 UTC 解析，年度不一致 {consistency['timezone']['year_mismatch']} 条 |")
    lines.append(f"| 有商品元数据的评论 | {reviews['has_meta']:,}（{consistency['join']['coverage']:.2%}） |")
    lines.append(f"| 无商品元数据的评论 | {reviews['without_meta']:,} |")
    lines.append(f"| 连接基数 | 评论 N : 1 商品；行数未因连接膨胀 |")
    lines.append(f"| 商品日期缺失导致的天数缺失 | {reviews['days_since_null_share']:.2%} |")
    lines.append("")
    lines.append("### 类型")
    lines.append("")
    lines.append("| 字段 | dtype |")
    lines.append("|---|---|")
    for column, dtype in consistency["types"]["review"].items():
        lines.append(f"| 评论.{column} | {dtype} |")
    for column, dtype in consistency["types"]["meta"].items():
        lines.append(f"| 商品.{column} | {dtype} |")
    lines.append("")
    lines.append("### 编码与脏数据")
    lines.append("")
    lines.append("| 检查项 | 数量 |")
    lines.append("|---|---|")
    lines.append(f"| 主品类含 `&amp;` | {meta['main_cat_with_amp']:,} |")
    lines.append(f"| 主品类含 `</span>` | {meta['main_cat_with_span']:,} |")
    lines.append(f"| 主品类字面值 `None` | {meta['main_cat_literal_none']:,} |")
    lines.append(f"| 类目列表为空 | {meta['category_empty']:,} |")
    lines.append(f"| 类目列表含 `</span>` 残片 | {meta['category_with_span']:,} |")
    lines.append(f"| 类目列表含 `&amp;` | {meta['category_with_amp']:,} |")
    lines.append(f"| 原始价格为空的商品 | {meta['price_raw_empty']:,} |")
    lines.append(f"| 原始价格以 `<div` 开头 | {meta['price_raw_div']:,} |")
    lines.append(f"| 原始价格以 `<span` 开头 | {meta['price_raw_span']:,} |")
    lines.append(f"| 原始价格以 `<script` 开头 | {meta['price_raw_script']:,} |")
    lines.append(f"| 可解析价格 | {meta['price_notnull']:,} |")
    lines.append(f"| 价格为 0 | {meta['price_zero']:,} |")
    lines.append(f"| 价格为负 | {meta['price_negative']:,} |")
    lines.append(f"| 共购列表非空 | {meta['also_buy_nonempty']:,} |")
    lines.append("")
    lines.append("### 断言")
    lines.append("")
    for item in consistency["assertions"]:
        lines.append(f"- {'通过' if item['passed'] else '失败'}：{item['name']}")
    lines.append("")
    lines.append("## 缺失填补策略对比")
    lines.append("")
    lines.append(
        f"目标为商品价格，取价格大于 0 的 {imputation['n_observed_positive']:,} 件商品，"
        f"按主品类分层抽 20% 做测试（{imputation['n_test']:,} 件），其余训练。"
    )
    lines.append("")
    lines.append("| 策略 | 说明 | MAE | RMSE | MedAE | R² |")
    lines.append("|---|---|---|---|---|---|")
    for item in imputation["strategies"]:
        lines.append(
            f"| {item['name']} | {item['detail']} | {item['mae']:.2f} | "
            f"{item['rmse']:.2f} | {item['medae']:.2f} | {item['r2']:.4f} |"
        )
    lines.append("")
    lines.append(
        f"按 MAE 最佳的是 **{imputation['best']}**。但测试集来自已观测价格，"
        "不能代表 83% 未观测价格的缺失机制；它只能说明“在已有价格内部，哪种回填更稳”。"
    )
    lines.append("")
    lines.append("## 异常值检测对比")
    lines.append("")
    lines.append("| 方法 | 范围 | 阈值 | 检出数 | 占比 |")
    lines.append("|---|---|---|---|---|")
    price_iqr = outliers["price_iqr"]
    lines.append(
        f"| 价格 IQR | 商品，price>0 | 上界 {price_iqr['upper']:.2f}（下界为负，无低价异常） | "
        f"{price_iqr['above']:,} | {price_iqr['above'] / price_iqr['n']:.2%} |"
    )
    len_iqr = outliers["review_len_iqr"]
    lines.append(
        f"| 评论长度 IQR | 全量评论 | 上界 {len_iqr['upper']:.0f}（下界为负） | "
        f"{len_iqr['above']:,} | {len_iqr['above'] / len_iqr['n']:.2%} |"
    )
    zscore = outliers["review_len_zscore"]
    lines.append(
        f"| 评论长度 z-score | 全量评论 | \\|z\\| > 3（均值 {zscore['mean']:.0f}，标准差 {zscore['std']:.0f}） | "
        f"{zscore['flagged']:,} | {zscore['share']:.2%} |"
    )
    forest = outliers["isolation_forest"]
    lines.append(
        f"| IsolationForest | {forest['n']:,} 条样本 | contamination=0.01 | "
        f"{forest['flagged']:,} | {forest['share']:.2%} |"
    )
    lines.append("")
    lines.append(
        f"IsolationForest 与长度 IQR 在同一份样本上的重叠为 {forest['overlap_with_iqr']:,} 条，"
        "两种方法关注的异常并不完全一致：IQR 只看单变量尾部，孤立森林会结合票数与图片数。"
    )
    lines.append("")
    lines.append("## 处理建议")
    lines.append("")
    for item in outliers["treatment"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 复现")
    lines.append("")
    lines.append("```bash")
    lines.append("python src/governance.py --force")
    lines.append("```")
    lines.append("")
    hardware = report["hardware"]
    lines.append(
        f"耗时 {report['seconds']:.1f} 秒；环境为 {hardware['platform']}，"
        f"{hardware['cpu_count']} 核，{hardware['ram_gb']} GB 内存，"
        f"Python {hardware['python']}，pyarrow {hardware['pyarrow']}。"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="数据治理验证与报告生成")
    parser.add_argument("--force", action="store_true", help="覆盖已有报告")
    args = parser.parse_args()

    for path in (REPORT_JSON, REPORT_MD):
        if path.exists() and not args.force:
            raise SystemExit(f"报告已存在，加 --force 覆盖: {path}")

    started = time.perf_counter()
    features = _load_features()
    meta = _load_meta()

    consistency = _consistency(features, meta)
    imputation = _imputation(meta)
    outliers = _outliers(features)

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seconds": round(time.perf_counter() - started, 2),
        "hardware": _hardware(),
        "consistency": consistency,
        "imputation": imputation,
        "outliers": outliers,
    }

    TABLES.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text(_markdown(report), encoding="utf-8")

    reviews = consistency["reviews"]
    print(f"评论 {reviews['rows']:,} 行，商品 {consistency['meta']['rows']:,} 行")
    print(f"重复复合键 {reviews['duplicate_key_rows']:,} 行，跨折 asin {reviews['asin_cross_fold']} 个")
    print(f"填补策略 {len(imputation['strategies'])} 组，最佳 {imputation['best']}")
    print(f"异常值：价格 IQR 检出 {outliers['price_iqr']['below'] + outliers['price_iqr']['above']:,} 件，"
          f"评论长度 z-score 检出 {outliers['review_len_zscore']['flagged']:,} 条")
    print(f"报告已写入 {REPORT_JSON} 与 {REPORT_MD}")


if __name__ == "__main__":
    main()
