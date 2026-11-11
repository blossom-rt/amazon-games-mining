"""P3 共购关联规则：also_buy 购物篮上的 Apriori 与 FP-Growth。

只使用 scope.parquet 中 in_scope=True 的商品。每个商品一个 transaction：
{自身 asin} 加上 also_buy 中仍在范围内的商品，去重后长度至少为 2。
同一份布尔矩阵、同一组阈值分别跑 Apriori 与 FP-Growth，比较规则集与耗时。

用法：
    python src/rules.py --force
    python src/rules.py --min-support 0.01 --min-confidence 0.3 --force
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

try:
    from .config import (
        DATA_PROCESSED,
        DEFAULT_MAX_BASKET_LEN,
        DEFAULT_MIN_CONFIDENCE,
        DEFAULT_MIN_SUPPORT,
        FIGURES,
        FONT_CANDIDATES,
        SEED,
        TABLES,
    )
    from .parse import _hardware
except ImportError:
    from config import (
        DATA_PROCESSED,
        DEFAULT_MAX_BASKET_LEN,
        DEFAULT_MIN_CONFIDENCE,
        DEFAULT_MIN_SUPPORT,
        FIGURES,
        FONT_CANDIDATES,
        SEED,
        TABLES,
    )
    from parse import _hardware

META_PARQUET = DATA_PROCESSED / "meta.parquet"
SCOPE_PARQUET = DATA_PROCESSED / "scope.parquet"
SUMMARY_JSON = TABLES / "rules_summary.json"
TOP_CSV = TABLES / "rules_top.csv"
THRESHOLDS_CSV = TABLES / "rules_thresholds.csv"
RUNTIME_CSV = TABLES / "rules_runtime.csv"

# 阈值敏感性实验：三组支持度与置信度组合，固定不变，便于复现
THRESHOLD_GRID = [(0.02, 0.15), (0.01, 0.2), (0.005, 0.1)]
ALGORITHMS = ("apriori", "fpgrowth")


def setup_fonts() -> None:
    plt.rcParams["font.sans-serif"] = FONT_CANDIDATES
    plt.rcParams["axes.unicode_minus"] = False


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def short(text: str, limit: int = 20) -> str:
    text = "" if not isinstance(text, str) else text.strip()
    if not text:
        return "(无标题)"
    return text if len(text) <= limit else text[: limit - 1] + "…"


def load_data():
    meta = pq.read_table(
        META_PARQUET, columns=["asin", "title", "also_buy"]
    ).to_pandas()
    scope = pq.read_table(SCOPE_PARQUET, columns=["asin", "in_scope"]).to_pandas()
    assert scope["asin"].is_unique, "scope 表 asin 不唯一"
    in_scope = set(scope.loc[scope["in_scope"], "asin"])
    assert len(in_scope) > 30_000, "范围内商品过少"
    return meta, in_scope


def build_transactions(meta, in_scope):
    titles = {
        asin: (title if isinstance(title, str) else "")
        for asin, title in zip(meta["asin"], meta["title"])
    }
    transactions = []
    dropped = 0
    for asin, also_buy in zip(meta["asin"], meta["also_buy"]):
        if asin not in in_scope:
            continue
        items = [asin]
        seen = {asin}
        for other in also_buy:
            if len(items) >= DEFAULT_MAX_BASKET_LEN:
                break
            if other in seen:
                continue
            if other not in in_scope:
                dropped += 1
                continue
            seen.add(other)
            items.append(other)
        if len(items) >= 2:
            transactions.append(items)
    return transactions, titles, dropped


def encode(transactions, min_support):
    n = len(transactions)
    counts = Counter(item for txn in transactions for item in txn)
    min_count = max(1, int(np.ceil(min_support * n)))
    items = sorted(item for item, count in counts.items() if count >= min_count)
    position = {item: j for j, item in enumerate(items)}
    matrix = np.zeros((n, len(items)), dtype=bool)
    for row, txn in enumerate(transactions):
        for item in txn:
            col = position.get(item)
            if col is not None:
                matrix[row, col] = True
    return pd.DataFrame(matrix, columns=items), counts


def mine(frame, algorithm, min_support, min_confidence, max_len, n_transactions):
    started = time.perf_counter()
    if frame.shape[1] == 0:
        return pd.DataFrame(), 0, time.perf_counter() - started
    itemsets = (
        apriori(frame, min_support=min_support, use_colnames=True, max_len=max_len)
        if algorithm == "apriori"
        else fpgrowth(frame, min_support=min_support, use_colnames=True, max_len=max_len)
    )
    if itemsets.empty:
        return pd.DataFrame(), 0, time.perf_counter() - started
    n_itemsets = int((itemsets["itemsets"].map(len) >= 2).sum())
    if n_itemsets == 0:
        return pd.DataFrame(), 0, time.perf_counter() - started
    rules = association_rules(
        itemsets,
        num_itemsets=n_transactions,
        metric="confidence",
        min_threshold=min_confidence,
    )
    return rules, n_itemsets, time.perf_counter() - started


def clean_rules(rules, n_transactions, titles):
    if rules.empty:
        empty = pd.DataFrame(
            columns=[
                "antecedents", "consequents", "support", "confidence", "lift",
                "antecedent", "consequent", "count", "pair_key",
                "antecedent_titles", "consequent_title",
            ]
        )
        empty.attrs["trivial_removed"] = 0
        empty.attrs["lift_removed"] = 0
        return empty
    trivial_mask = (
        rules["antecedents"].map(len).eq(0)
        | rules["consequents"].map(len).eq(0)
    )
    trivial_removed = int(trivial_mask.sum())
    rules = rules.loc[~trivial_mask]
    before_lift = len(rules)
    rules = rules.loc[rules["lift"] > 1.0].copy()
    lift_removed = before_lift - len(rules)
    rules["antecedent"] = rules["antecedents"].map(lambda s: "|".join(sorted(s)))
    rules["consequent"] = rules["consequents"].map(lambda s: "|".join(sorted(s)))
    rules["count"] = (rules["support"] * n_transactions).round().astype(int)
    rules["pair_key"] = [
        " <=> ".join(sorted([a, b]))
        for a, b in zip(rules["antecedent"], rules["consequent"])
    ]
    rules = (
        rules.sort_values(["confidence", "lift"], ascending=False)
        .drop_duplicates("pair_key")
        .sort_values(["lift", "confidence"], ascending=False)
    )
    rules["antecedent_titles"] = rules["antecedent"].map(
        lambda s: " | ".join(titles.get(item, "") for item in s.split("|"))
    )
    rules["consequent_title"] = rules["consequent"].map(
        lambda item: titles.get(item, "")
    )
    rules.attrs["trivial_removed"] = trivial_removed
    rules.attrs["lift_removed"] = lift_removed
    return rules


def plot_basket_length(lengths, path):
    plt.figure(figsize=(7, 4))
    plt.hist(lengths, bins=np.arange(2, min(int(lengths.max()), 60) + 2) - 0.5)
    plt.yscale("log")
    plt.xlabel("购物篮长度（商品数）")
    plt.ylabel("商品数（对数刻度）")
    plt.title("游戏本体共购篮长度分布")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rule_scatter(cleaned, path):
    sample = cleaned.sample(n=min(5_000, len(cleaned)), random_state=SEED)
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        sample["support"], sample["confidence"], c=sample["lift"],
        s=10, alpha=0.5, cmap="viridis",
    )
    plt.colorbar(scatter, label="提升度 lift")
    plt.xlabel("支持度 support")
    plt.ylabel("置信度 confidence")
    plt.title("规则的支持度、置信度与提升度")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_top_lift(top, titles, path):
    rows = top.head(15).iloc[::-1]
    labels = []
    for row in rows.itertuples(index=False):
        ante = row.antecedent.split("|")[0]
        cons = row.consequent.split("|")[0]
        labels.append(f"{short(titles.get(ante, ante), 16)} → {short(titles.get(cons, cons), 16)}")
    plt.figure(figsize=(9, 6))
    plt.barh(labels, rows["lift"].to_numpy(), color="#4c72b0")
    plt.xlabel("提升度 lift")
    plt.title("提升度最高的 15 条共购规则")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_runtime(runtime_rows, path):
    frame = pd.DataFrame(runtime_rows)
    frame["label"] = frame.apply(
        lambda r: f"s={r['min_support']}\nc={r['min_confidence']}", axis=1
    )
    labels = frame["label"].drop_duplicates().tolist()
    width = 0.38
    x = np.arange(len(labels))
    plt.figure(figsize=(8, 4.5))
    for offset, algorithm in zip((-width / 2, width / 2), ALGORITHMS):
        part = frame.loc[frame["algorithm"] == algorithm]
        part = part.drop_duplicates("label").set_index("label").reindex(labels)
        bars = plt.bar(x + offset, part["seconds"], width=width, label=algorithm)
        for rect, value in zip(bars, part["seconds"].to_numpy()):
            if np.isfinite(value):
                plt.text(
                    rect.get_x() + rect.get_width() / 2,
                    max(float(value), 0.01) * 1.15,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    plt.xticks(x, labels)
    plt.yscale("log")
    plt.ylabel("运行时间（秒，对数刻度）")
    plt.title("Apriori 与 FP-Growth 在三组阈值下的耗时")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_lift_distribution(cleaned, path):
    plt.figure(figsize=(7, 4))
    plt.hist(cleaned["lift"], bins=60)
    plt.yscale("log")
    plt.xlabel("提升度 lift")
    plt.ylabel("规则数（对数刻度）")
    plt.title("清洗后规则的提升度分布")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_top_items(counts, titles, path):
    top = counts.most_common(20)
    labels = [short(titles.get(item, item), 24) for item, _ in top][::-1]
    values = [count for _, count in top][::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(labels, values, color="#55a868")
    plt.xlabel("出现在多少个购物篮中")
    plt.title("出现频次最高的 20 件游戏商品")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def safe_plot(func, *args) -> str | None:
    try:
        func(*args)
        return None
    except Exception as error:
        print(f"画图失败 {func.__name__}: {error}")
        return f"{func.__name__}: {error}"


def main() -> None:
    parser = argparse.ArgumentParser(description="also_buy 共购关联规则")
    parser.add_argument("--min-support", type=float, default=DEFAULT_MIN_SUPPORT)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--max-len", type=int, default=2, help="项集最大长度，2 为商品对")
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if SUMMARY_JSON.exists() and not args.force:
        raise SystemExit(f"输出已存在，加 --force 覆盖: {SUMMARY_JSON}")
    assert 0 < args.min_support < 1, "min-support 必须在 (0, 1)"
    assert 0 <= args.min_confidence <= 1, "min-confidence 必须在 [0, 1]"
    assert args.max_len >= 2, "max-len 至少为 2"

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    started = time.perf_counter()

    meta, in_scope = load_data()
    transactions, titles, dropped = build_transactions(meta, in_scope)
    n = len(transactions)
    assert n > 0, "没有构造出任何购物篮"
    lengths = np.array([len(txn) for txn in transactions])
    print(f"购物篮 {n:,} 个，最长 {lengths.max()}，中位 {np.median(lengths):.0f}")
    print(f"范围外商品被丢弃 {dropped:,} 次")

    frame, counts = encode(transactions, args.min_support)
    print(f"参与挖掘的频繁商品 {frame.shape[1]:,} 个")

    results = {}
    for algorithm in ALGORITHMS:
        rules, n_itemsets, seconds = mine(
            frame, algorithm, args.min_support, args.min_confidence,
            args.max_len, n,
        )
        results[algorithm] = {"rules": rules, "itemsets": n_itemsets, "seconds": seconds}
        print(f"{algorithm}: {len(rules):,} 条规则，{n_itemsets:,} 个项集，{seconds:.1f} 秒")

    base = results["apriori"]["rules"]
    other = results["fpgrowth"]["rules"]
    assert len(base) > 0, "没有挖到规则，降低 min_support 或 min_confidence"

    def signatures(rules):
        return {
            (frozenset(a), frozenset(c))
            for a, c in zip(rules["antecedents"], rules["consequents"])
        }

    same_rule_set = signatures(base) == signatures(other)
    assert same_rule_set, "Apriori 与 FP-Growth 规则集不一致"

    cleaned = clean_rules(base, n, titles)
    assert len(cleaned) > 0, "清洗后没有规则"
    trivial_remaining = int(
        (cleaned["antecedents"].map(len).eq(0)
         | cleaned["consequents"].map(len).eq(0)).sum()
    )
    assert trivial_remaining == 0, "清洗后仍有平凡规则"
    rule_items = set().union(
        *[set(a) | set(c) for a, c in zip(base["antecedents"], base["consequents"])]
    )
    scope_only = rule_items <= in_scope
    assert scope_only, "规则里出现了范围外商品"

    top = cleaned.head(args.top).reset_index(drop=True)
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    top[
        [
            "rank", "antecedent", "consequent", "support", "confidence",
            "lift", "count", "antecedent_titles", "consequent_title",
        ]
    ].to_csv(TOP_CSV, index=False, encoding="utf-8-sig")

    threshold_rows = []
    runtime_rows = []
    for min_support, min_confidence in THRESHOLD_GRID:
        grid_frame, _ = encode(transactions, min_support)
        row = {"min_support": min_support, "min_confidence": min_confidence}
        for algorithm in ALGORITHMS:
            rules, n_itemsets, seconds = mine(
                grid_frame, algorithm, min_support, min_confidence,
                args.max_len, n,
            )
            deduped = clean_rules(rules, n, titles)
            row[f"{algorithm}_seconds"] = round(seconds, 2)
            runtime_rows.append(
                {
                    "algorithm": algorithm,
                    "min_support": min_support,
                    "min_confidence": min_confidence,
                    "n_transactions": n,
                    "n_itemsets": n_itemsets,
                    "n_rules": len(rules),
                    "seconds": round(seconds, 2),
                }
            )
            if algorithm == "apriori":
                row["n_rules"] = len(rules)
                row["n_rules_dedup"] = len(deduped)
                if deduped.empty:
                    row["median_lift"] = None
                    row["max_lift"] = None
                else:
                    row["median_lift"] = round(float(deduped["lift"].median()), 4)
                    row["max_lift"] = round(float(deduped["lift"].max()), 4)
        threshold_rows.append(row)
        print(f"阈值 support={min_support} confidence={min_confidence} 完成")

    thresholds = pd.DataFrame(threshold_rows)[
        [
            "min_support", "min_confidence", "n_rules", "n_rules_dedup",
            "median_lift", "max_lift", "apriori_seconds", "fpgrowth_seconds",
        ]
    ]
    thresholds.to_csv(THRESHOLDS_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(runtime_rows).to_csv(RUNTIME_CSV, index=False, encoding="utf-8-sig")

    figure_errors = [
        error
        for error in [
            safe_plot(plot_basket_length, lengths, FIGURES / "fig11_basket_length.png"),
            safe_plot(plot_rule_scatter, cleaned, FIGURES / "fig12_rule_scatter.png"),
            safe_plot(plot_top_lift, top, titles, FIGURES / "fig13_top_lift.png"),
            safe_plot(plot_runtime, runtime_rows, FIGURES / "fig14_runtime.png"),
            safe_plot(
                plot_lift_distribution, cleaned,
                FIGURES / "fig15_lift_distribution.png",
            ),
            safe_plot(plot_top_items, counts, titles, FIGURES / "fig16_top_items.png"),
        ]
        if error
    ]

    assertions = {
        "same_rule_set": bool(same_rule_set),
        "non_empty": bool(len(cleaned) > 0),
        "scope_only": bool(scope_only),
        "support_in_range": bool(base["support"].between(0, 1).all()),
        "confidence_in_range": bool(base["confidence"].between(0, 1).all()),
        "runtime_recorded": all(row["seconds"] >= 0 for row in runtime_rows),
        "no_trivial_rules": bool(trivial_remaining == 0),
    }
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "seed": SEED,
        "input": {
            "transactions": n,
            "baskets_ge2": n,
            "unique_asins": len(counts),
            "max_basket_len": int(lengths.max()),
            "basket_len_median": float(np.median(lengths)),
            "out_of_scope_items_dropped": dropped,
        },
        "params": {
            "min_support": args.min_support,
            "min_confidence": args.min_confidence,
            "max_len": args.max_len,
            "max_basket_len": DEFAULT_MAX_BASKET_LEN,
            "threshold_grid": THRESHOLD_GRID,
        },
        "runtime_seconds": {
            "apriori": round(results["apriori"]["seconds"], 2),
            "fpgrowth": round(results["fpgrowth"]["seconds"], 2),
        },
        "rule_counts": {
            "apriori": len(base),
            "fpgrowth": len(other),
            "after_dedup": len(cleaned),
            "trivial_removed": int(cleaned.attrs.get("trivial_removed", 0)),
            "lift_filtered": int(cleaned.attrs.get("lift_removed", 0)),
        },
        "assertions": assertions,
        "figure_errors": figure_errors,
        "hardware": _hardware(),
        "outputs": {
            "top": str(TOP_CSV),
            "thresholds": str(THRESHOLDS_CSV),
            "runtime": str(RUNTIME_CSV),
        },
    }
    SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"完成，用时 {time.perf_counter() - started:.1f} 秒，摘要 {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
