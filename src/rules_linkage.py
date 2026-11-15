"""规则与业务联动：同系列/跨系列、品类对与可执行组合。

只读 results/tables/rules_top.csv 与 meta.parquet，不重新挖规则。
输出：
    results/tables/rules_linkage.csv   跨系列规则（按 support 排序）
    results/tables/rules_linkage.json  同系列比例、品类对、头部组合

用法：
    python src/rules_linkage.py
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq

try:
    from .config import DATA_PROCESSED, TABLES
except ImportError:
    from config import DATA_PROCESSED, TABLES

META_PARQUET = DATA_PROCESSED / "meta.parquet"

RULES_CSV = TABLES / "rules_all.csv"
OUT_CSV = TABLES / "rules_linkage.csv"
OUT_JSON = TABLES / "rules_linkage.json"


def leaf_label(value) -> str:
    """把类目路径压成可读叶子，过滤掉商品描述句。"""
    if value is None or not hasattr(value, "__len__"):
        return "(unknown)"
    names = [str(item).strip() for item in list(value) if str(item).strip()]
    for preferred in ("Games", "Accessories", "Consoles", "PC", "Mac"):
        if preferred in names:
            return preferred
    for name in reversed(names):
        if len(name) <= 30 and len(name.split()) <= 4:
            return name
    return "(unknown)"


def build_categories() -> pd.DataFrame:
    meta = pq.read_table(META_PARQUET, columns=["asin", "category", "main_cat"]).to_pandas()
    meta["leaf"] = [leaf_label(value) for value in meta["category"]]
    meta["main_cat"] = meta["main_cat"].fillna("(unknown)")
    return meta[["asin", "leaf", "main_cat"]].drop_duplicates("asin")


def main() -> None:
    rules = pd.read_csv(RULES_CSV)
    categories = build_categories()
    for side in ["antecedent", "consequent"]:
        mapping = categories.rename(
            columns={"asin": side, "leaf": f"{side}_leaf", "main_cat": f"{side}_main_cat"}
        )
        rules = rules.merge(mapping, on=side, how="left")
    for column in ["antecedent_leaf", "consequent_leaf", "antecedent_main_cat", "consequent_main_cat"]:
        rules[column] = rules[column].fillna("(unknown)")

    rules["same_leaf"] = rules["antecedent_leaf"] == rules["consequent_leaf"]
    rules["category_pair"] = rules["antecedent_leaf"] + " → " + rules["consequent_leaf"]
    rules["download"] = (
        rules["antecedent_titles"].fillna("").str.contains(r"\[Download\]", regex=True)
        | rules["consequent_title"].fillna("").str.contains(r"\[Download\]", regex=True)
    )

    cross = rules.loc[~rules["same_series"]].copy()
    cross = cross.sort_values(["support", "confidence"], ascending=False)
    columns = [
        "rank", "antecedent", "consequent", "antecedent_titles", "consequent_title",
        "antecedent_leaf", "consequent_leaf", "category_pair", "support", "confidence",
        "lift", "conviction", "count", "same_leaf", "download",
    ]
    cross[columns].head(50).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    pair_counts = (
        rules.groupby("category_pair", dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(15)
    )
    top_support = cross.head(15)[columns].to_dict("records")
    top_lift = cross.sort_values("lift", ascending=False).head(15)[columns].to_dict("records")

    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_rules": int(len(rules)),
        "same_series": int(rules["same_series"].sum()),
        "same_series_share": round(float(rules["same_series"].mean()), 4),
        "same_leaf": int(rules["same_leaf"].sum()),
        "same_leaf_share": round(float(rules["same_leaf"].mean()), 4),
        "download_rules": int(rules["download"].sum()),
        "cross_series_support": {
            "mean": round(float(cross["support"].mean()), 5),
            "median": round(float(cross["support"].median()), 5),
            "max": round(float(cross["support"].max()), 5),
        },
        "top_category_pairs": [
            {"pair": str(pair), "rules": int(count)} for pair, count in pair_counts.items()
        ],
        "top_cross_series_by_support": top_support,
        "top_cross_series_by_lift": top_lift,
        "outputs": {"csv": str(OUT_CSV)},
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in [
        "n_rules", "same_series", "same_series_share", "same_leaf_share",
        "download_rules", "cross_series_support", "top_category_pairs",
    ]}, ensure_ascii=False, indent=2))
    print(cross[columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
