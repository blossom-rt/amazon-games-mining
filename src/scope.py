"""P2.5 品类边界：判定商品是否属于“游戏本体”。

规则只使用商品类目路径与标题，不读取评分，避免把目标信息带进样本筛选。
脚本在判定完成后读取 overall，仅用于汇总范围内外正类率，不参与规则。

判定顺序：
1. 类目为空，或根节点不是 Video Games：排除；
2. 路径含 Accessories、硬件叶子或 Downloadable Content：排除；
3. 路径含 Games / Digital Games 节点：保留；
4. 仅两层且第二层是平台（如 Video Games > PC）：保留，但标题含硬件词时排除；
5. 其余路径：排除，留给人工复核。

用法：
    python src/scope.py
    python src/scope.py --force
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq

try:
    from .config import DATA_PROCESSED, SCOPE_PARQUET, SEED, TABLES
    from .parse import _hardware
except ImportError:
    from config import DATA_PROCESSED, SCOPE_PARQUET, SEED, TABLES
    from parse import _hardware

META_PARQUET = DATA_PROCESSED / "meta.parquet"
REVIEWS_PARQUET = DATA_PROCESSED / "reviews.parquet"


SOFTWARE_NODES = {"games", "digital games"}

HARDWARE_LEAVES = {
    "accessories",
    "accessory",
    "accessory kits",
    "adapters",
    "bags & cases",
    "batteries",
    "batteries & chargers",
    "cables",
    "cables & adapters",
    "cases & storage",
    "chargers",
    "controllers",
    "cooling systems",
    "dance pads",
    "faceplates",
    "faceplates, protectors & skins",
    "figures",
    "flight controls",
    "gamepads & standard controllers",
    "gaming keyboards",
    "gaming mice",
    "headsets",
    "interactive gaming figures",
    "joysticks",
    "memory",
    "microphones",
    "mounts, brackets & stands",
    "racing wheels",
    "remotes",
    "screen protectors",
    "skins",
    "stands",
    "stylus pens",
    "thumb grips",
}

HARDWARE_WORDS = (
    "accessor",
    "adapter",
    "battery",
    "cable",
    "case",
    "charger",
    "console",
    "controller",
    "cover",
    "dock",
    "figure",
    "headset",
    "hub",
    "joystick",
    "keyboard",
    "microphone",
    "mount",
    "mouse",
    "remote",
    "skin",
    "speaker",
    "stand",
    "stylus",
    "wheel",
)

HARDWARE_TITLE_RE = re.compile(
    r"\b(?:"
    r"case|cover|skin|protector|screen|cable|cord|charger|battery|"
    r"controller|gamepad|headset|mouse|keyboard|stand|mount|dock|stylus|"
    r"adapter|repair|replacement|memory card|dance pad|console|"
    r"headphone|microphone|joystick|remote"
    r")\b",
    re.IGNORECASE,
)

IN_SCOPE_REASONS = {"software_node", "platform_only"}


def effective_path(value) -> list[str]:
    """截掉类目路径末尾的 HTML 残片与商品描述句。"""
    try:
        items = list(value)
    except TypeError:
        return []
    path: list[str] = []
    for item in items:
        text = re.sub(r"\s+", " ", str(item)).strip()
        if not text:
            continue
        if "</span" in text or text.lower() in {"none", "null"}:
            break
        if len(text) > 90:
            break
        path.append(text)
    return path


def classify_products(meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asin, title, category in meta[["asin", "title", "category"]].itertuples(
        index=False, name=None
    ):
        path = effective_path(category)
        lower = [segment.lower() for segment in path]
        title_text = title if isinstance(title, str) else ""
        title_hardware = bool(HARDWARE_TITLE_RE.search(title_text))
        has_accessories = any("accessor" in segment for segment in lower)
        has_hardware_leaf = bool(path) and (
            lower[-1] in HARDWARE_LEAVES
            or any(word in lower[-1] for word in HARDWARE_WORDS)
        )
        has_dlc = any("downloadable content" in segment for segment in lower)
        has_software_node = any(segment in SOFTWARE_NODES for segment in lower)
        platform_only = (
            len(path) == 2 and lower[0] == "video games" and not title_hardware
        )

        if not path:
            reason = "empty_category"
        elif lower[0] != "video games":
            reason = "non_video_games"
        elif has_accessories:
            reason = "accessories"
        elif has_hardware_leaf:
            reason = "hardware_leaf"
        elif has_dlc:
            reason = "dlc"
        elif has_software_node:
            reason = "software_node"
        elif platform_only:
            reason = "platform_only"
        elif len(path) == 2 and title_hardware:
            reason = "hardware_title"
        else:
            reason = "no_software_node"

        rows.append(
            {
                "asin": asin,
                "in_scope": reason in IN_SCOPE_REASONS,
                "reason": reason,
                "path_depth": len(path),
                "path_text": " > ".join(path),
            }
        )
    return pd.DataFrame(rows)


def build_scope(force: bool) -> dict:
    if SCOPE_PARQUET.exists() and not force:
        raise SystemExit(f"输出已存在，加 --force 覆盖: {SCOPE_PARQUET}")
    TABLES.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    meta = pq.read_table(META_PARQUET).to_pandas()
    scope = classify_products(meta)
    reviews = pq.read_table(REVIEWS_PARQUET, columns=["asin", "overall"]).to_pandas()
    checked = reviews.merge(scope[["asin", "in_scope"]], on="asin", how="left")
    checked["in_scope"] = checked["in_scope"].fillna(False).astype(bool)

    scope.to_parquet(SCOPE_PARQUET, index=False, compression="zstd")

    products_in = int(scope["in_scope"].sum())
    reviews_in = int(checked["in_scope"].sum())
    reason_counts = scope["reason"].value_counts().to_dict()
    main_cat = (
        meta.merge(scope[["asin", "in_scope"]], on="asin", how="left")
        .loc[lambda frame: frame["in_scope"], "main_cat"]
        .fillna("(缺失)")
        .value_counts()
        .head(10)
    )
    top_paths = (
        scope.loc[scope["in_scope"], "path_text"].value_counts().head(15)
    )
    examples = {}
    labelled = meta.merge(
        scope[["asin", "reason", "path_text"]], on="asin", how="left"
    )
    for reason in scope["reason"].unique():
        sample = labelled.loc[
            labelled["reason"] == reason, ["title", "path_text"]
        ].head(3)
        examples[reason] = [
            {"title": row.title, "path": row.path_text}
            for row in sample.itertuples(index=False)
        ]

    assert scope["asin"].is_unique, "scope 的 asin 不唯一"
    assert len(scope) == len(meta), "scope 与商品表行数不一致"
    assert products_in >= 30_000, f"在范围内商品过少: {products_in}"
    assert reviews_in >= 1_000_000, f"在范围内评论过少: {reviews_in}"
    assert set(scope.loc[scope["in_scope"], "reason"]) <= IN_SCOPE_REASONS
    assert not scope.loc[scope["in_scope"], "reason"].eq("hardware_title").any()

    elapsed = time.perf_counter() - started
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seconds": round(elapsed, 1),
        "seed": SEED,
        "products_total": int(len(scope)),
        "products_in_scope": products_in,
        "products_out_scope": int((~scope["in_scope"]).sum()),
        "product_scope_share": round(products_in / len(scope), 4),
        "reviews_total": int(len(reviews)),
        "reviews_in_scope": reviews_in,
        "reviews_out_scope": int((~checked["in_scope"]).sum()),
        "review_scope_share": round(reviews_in / len(reviews), 4),
        "positive_rate_in_scope": round(
            float(checked.loc[checked["in_scope"], "overall"].ge(4).mean()), 4
        ),
        "positive_rate_out_scope": round(
            float(checked.loc[~checked["in_scope"], "overall"].ge(4).mean()), 4
        ),
        "reasons": {str(key): int(value) for key, value in reason_counts.items()},
        "in_scope_main_cat": {str(key): int(value) for key, value in main_cat.items()},
        "top_in_scope_paths": {
            str(key): int(value) for key, value in top_paths.items()
        },
        "examples": examples,
        "hardware": _hardware(),
    }
    (TABLES / "scope_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"商品 {len(scope):,} 件，在范围 {products_in:,} 件")
    print(f"评论 {len(reviews):,} 条，在范围 {reviews_in:,} 条")
    print("排除原因", summary["reasons"])
    print(f"范围表已写入 {SCOPE_PARQUET}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="判定商品是否属于游戏本体")
    parser.add_argument("--force", action="store_true", help="覆盖已有输出")
    args = parser.parse_args()
    build_scope(force=args.force)


if __name__ == "__main__":
    main()
