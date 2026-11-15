"""分类误差的分群统计：评分档、主品类、verified、年份与标签组。

使用 `classify_decision_oof.parquet` 的 20 万条 OOF 概率（线性逻辑回归，
按 asin 5 折），在 0.5 阈值下统计每组的样本量、错误率、准确率与宏 F1。
"""

from __future__ import annotations

import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

try:
    from .config import FIGURES, FONT_CANDIDATES, TABLES
except ImportError:
    from config import FIGURES, FONT_CANDIDATES, TABLES

IN_OOF = TABLES / "classify_decision_oof.parquet"
OUT_CSV = TABLES / "classify_subgroups.csv"
OUT_JSON = TABLES / "classify_subgroups.json"
OUT_FIG = FIGURES / "fig49_subgroup_errors.png"


def group_stats(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, part in frame.groupby(column, dropna=False, observed=True):
        if len(part) == 0:
            continue
        y_true = part["label"].to_numpy()
        y_pred = part["prediction"].to_numpy()
        rows.append(
            {
                "dimension": column,
                "group": str(value),
                "n": int(len(part)),
                "positive_rate": float(y_true.mean()),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "error_rate": float(np.mean(y_true != y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "false_positive": int(np.sum((y_pred == 1) & (y_true == 0))),
                "false_negative": int(np.sum((y_pred == 0) & (y_true == 1))),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    plt.rcParams["font.sans-serif"] = FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False
    frame = pd.read_parquet(IN_OOF)
    frame["prediction"] = (frame["probability_calibrated"] >= 0.5).astype(int)
    frame["rating_group"] = np.where(frame["overall"] >= 4, "4–5 星", "1–3 星")
    frame["year_group"] = pd.cut(
        frame["review_year"],
        bins=[1996, 2010, 2014, 2018],
        labels=["1997–2010", "2011–2014", "2015–2018"],
    )

    top_cats = frame["main_cat_clean"].value_counts().head(6).index
    frame["main_cat_group"] = np.where(
        frame["main_cat_clean"].isin(top_cats), frame["main_cat_clean"].astype(str), "其他"
    )

    parts = []
    for column in ["rating_group", "overall", "main_cat_group", "verified", "year_group"]:
        parts.append(group_stats(frame, column))
    result = pd.concat(parts, ignore_index=True)
    result.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    overall = group_stats(frame.assign(all="全部"), "all").iloc[0].to_dict()
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(IN_OOF),
        "n_rows": int(len(frame)),
        "threshold": 0.5,
        "overall": overall,
        "groups": result.to_dict(orient="records"),
        "notes": [
            "20 万条 OOF 概率，线性逻辑回归、按 asin 5 折；",
            "误差率为 1-准确率；组内宏 F1 在只有单一类别的组会退化，需结合样本量看。",
        ],
        "outputs": {"csv": str(OUT_CSV), "figure": str(OUT_FIG)},
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    rating = result.loc[result["dimension"] == "overall"].copy()
    axes[0].bar(rating["group"], rating["error_rate"] * 100, color="#c44e52")
    axes[0].set_xlabel("评分档（星）")
    axes[0].set_ylabel("错误率（%）")
    axes[0].set_title("按评分档的分群错误率")
    axes[0].grid(axis="y", alpha=0.25)

    cats = result.loc[result["dimension"] == "main_cat_group"].sort_values("n", ascending=False)
    axes[1].barh(cats["group"], cats["error_rate"] * 100, color="#4c72b0")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("错误率（%）")
    axes[1].set_title("按主品类的分群错误率")
    axes[1].grid(axis="x", alpha=0.25)
    figure.suptitle("分类误差分群统计（20 万 OOF，阈值 0.5）")
    figure.tight_layout()
    figure.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(result.to_string(index=False))
    print(f"总体错误率 {overall['error_rate']:.4f}；写入 {OUT_CSV}")


if __name__ == "__main__":
    main()
