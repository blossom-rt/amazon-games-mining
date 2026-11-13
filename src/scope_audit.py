"""P2.5 边界复核：强硬件词候选与 Retro 缺失节点统计。

只生成复核清单，不自动改样本口径；若要收紧，先人工确认清单中的“同捆”和
“误录”，再决定是否把规则写回 scope.py。

用法：
    python src/scope_audit.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq

try:
    from .config import SEED, TABLES
except ImportError:
    from config import SEED, TABLES

META_PARQUET = "data/processed/meta.parquet"
SCOPE_PARQUET = "data/processed/scope.parquet"
FEATURES_PARQUET = "data/processed/features.parquet"

# 比 scope.py 的 HARDWARE_TITLE_RE 更窄：去掉 case/cover/stand 等常见游戏标题词，
# 只保留“作为商品主体的硬件名词”。
STRONG_HARDWARE = re.compile(
    r"\b(?:controller|gamepad|headset|console|memory card|charger|power cord|"
    r"adapter|stylus|dock|repair|replacement|dance pad|joystick|remote)\b",
    re.IGNORECASE,
)


def main() -> None:
    meta = pq.read_table(META_PARQUET).to_pandas()
    scope = pq.read_table(SCOPE_PARQUET).to_pandas()
    products = meta.merge(
        scope[["asin", "in_scope", "reason", "path_text"]], on="asin", how="left"
    )

    candidates = products.loc[
        products["in_scope"].fillna(False)
        & (products["reason"] == "software_node")
        & products["title"].fillna("").map(lambda text: bool(STRONG_HARDWARE.search(text)))
    ].copy()
    candidates["matched"] = candidates["title"].fillna("").map(
        lambda text: ", ".join(sorted(set(m.group(0).lower() for m in STRONG_HARDWARE.finditer(text))))
    )
    candidates["review_verdict"] = ""
    candidates["review_note"] = ""

    reviews = pq.read_table(
        "data/processed/reviews.parquet", columns=["asin"]
    ).to_pandas()
    review_counts = reviews.groupby("asin").size()
    candidates["n_reviews"] = candidates["asin"].map(review_counts).fillna(0).astype(int)
    candidates = candidates.sort_values("n_reviews", ascending=False)

    retro_mask = products["category"].astype(str).str.contains(
        "Retro Gaming", case=False, na=False
    )
    retro = products.loc[retro_mask].copy()
    retro_excluded = retro.loc[
        ~retro["in_scope"].fillna(False) & (retro["reason"] == "no_software_node")
    ].copy()
    retro_excluded["n_reviews"] = retro_excluded["asin"].map(review_counts).fillna(0).astype(int)
    retro_excluded = retro_excluded.sort_values("n_reviews", ascending=False)

    strict_products = int(products["in_scope"].fillna(False).sum() - len(candidates))
    in_scope_reviews = set(
        products.loc[products["in_scope"].fillna(False), "asin"]
    )
    strict_reviews = int(
        reviews["asin"].isin(in_scope_reviews - set(candidates["asin"])).sum()
    )

    output_columns = [
        "asin", "title", "main_cat", "path_text", "matched",
        "n_reviews", "review_verdict", "review_note",
    ]
    candidates[output_columns].to_csv(
        TABLES / "scope_audit_candidates.csv", index=False, encoding="utf-8-sig"
    )
    retro_excluded[
        ["asin", "title", "main_cat", "path_text", "n_reviews"]
    ].to_csv(TABLES / "scope_audit_retro.csv", index=False, encoding="utf-8-sig")

    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": SEED,
        "current_in_scope_products": int(products["in_scope"].fillna(False).sum()),
        "current_in_scope_reviews": int(
            reviews["asin"].isin(in_scope_reviews).sum()
        ),
        "strong_hardware_candidates": int(len(candidates)),
        "candidate_reviews": int(candidates["n_reviews"].sum()),
        "if_excluded_products": strict_products,
        "if_excluded_reviews": strict_reviews,
        "retro_products": int(len(retro)),
        "retro_no_software_node": int(len(retro_excluded)),
        "retro_no_software_node_reviews": int(retro_excluded["n_reviews"].sum()),
        "notes": [
            "强硬件词清单只用于人工复核，不自动改样本口径；",
            "同捆（游戏 + 手柄/主机）与纯硬件需要逐条判断；",
            "Retro 缺 Games 节点的商品无法确认是否游戏本体，保守排除。",
        ],
        "outputs": {
            "candidates": str(TABLES / "scope_audit_candidates.csv"),
            "retro": str(TABLES / "scope_audit_retro.csv"),
        },
    }
    (TABLES / "scope_audit_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(candidates[output_columns].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
