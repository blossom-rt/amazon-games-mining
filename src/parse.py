"""P0 数据解析：把两个 gzip JSON Lines 文件转成 parquet。

评论表保留评论文本与评分，供分类任务使用；商品表把嵌套列表拆成
parquet 的 list 列并按 asin 去重，供关联规则与聚类使用。

用法：
    python src/parse.py
    python src/parse.py --limit 5000 --force     # 冒烟测试
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import time
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    from .config import DATA_PROCESSED, META_FILE, REVIEW_FILE, TABLES
    from .io_utils import count_lines_gz, iter_jsonl_gz
except ImportError:
    from config import DATA_PROCESSED, META_FILE, REVIEW_FILE, TABLES
    from io_utils import count_lines_gz, iter_jsonl_gz

REVIEWS_OUT = DATA_PROCESSED / "reviews.parquet"
META_OUT = DATA_PROCESSED / "meta.parquet"
LOG_OUT = TABLES / "parse_log.json"

REVIEW_SCHEMA = pa.schema(
    [
        ("reviewerID", pa.string()),
        ("asin", pa.string()),
        ("reviewerName", pa.string()),
        ("overall", pa.float32()),
        ("verified", pa.bool_()),
        ("reviewTime", pa.string()),
        ("unixReviewTime", pa.int64()),
        ("reviewText", pa.string()),
        ("summary", pa.string()),
        ("style", pa.string()),
        ("vote", pa.int32()),
        ("image_count", pa.int32()),
    ]
)

META_SCHEMA = pa.schema(
    [
        ("asin", pa.string()),
        ("title", pa.string()),
        ("brand", pa.string()),
        ("price_raw", pa.string()),
        ("price", pa.float64()),
        ("main_cat", pa.string()),
        ("category", pa.list_(pa.string())),
        ("also_buy", pa.list_(pa.string())),
        ("also_view", pa.list_(pa.string())),
        ("feature", pa.list_(pa.string())),
        ("description", pa.string()),
        ("rank", pa.list_(pa.string())),
        ("imageURL", pa.list_(pa.string())),
        ("imageURLHighRes", pa.list_(pa.string())),
        ("tech1", pa.string()),
        ("tech2", pa.string()),
        ("fit", pa.string()),
        ("similar_item", pa.string()),
        ("date", pa.string()),
    ]
)

# 原始 price 里混有 HTML 片段，只接受以美元符号开头的金额
PRICE_RE = re.compile(r"^\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)")


def _str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _str_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]


def _parse_price(raw: str | None) -> float | None:
    if not raw:
        return None
    match = PRICE_RE.match(raw.strip())
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _new_columns(schema: pa.Schema) -> dict[str, list]:
    return {name: [] for name in schema.names}


def _flush(writer: pq.ParquetWriter, columns: dict[str, list], schema: pa.Schema) -> None:
    if not columns[schema.names[0]]:
        return
    table = pa.Table.from_pydict(columns, schema=schema)
    writer.write_table(table)
    for values in columns.values():
        values.clear()


def _style_text(style) -> str | None:
    if not isinstance(style, dict) or not style:
        return None
    parts = []
    for key, value in style.items():
        key = str(key).strip().rstrip(":").strip()
        parts.append(f"{key}={str(value).strip()}")
    return "; ".join(parts) or None


def _vote_count(vote) -> int | None:
    if isinstance(vote, int):
        return vote
    if isinstance(vote, str):
        digits = re.sub(r"\D", "", vote)
        return int(digits) if digits else None
    return None


def parse_reviews(
    out_path: Path,
    total_lines: int,
    chunk_size: int,
    limit: int | None,
    force: bool,
) -> dict:
    if out_path.exists() and not force:
        raise SystemExit(f"输出已存在，加 --force 覆盖: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"lines": 0, "empty": 0, "bad": 0}
    columns = _new_columns(REVIEW_SCHEMA)
    writer = pq.ParquetWriter(out_path, REVIEW_SCHEMA, compression="zstd")
    rows = 0
    rating_min = None
    rating_max = None
    null_asin = 0
    image_reviews = 0
    style_reviews = 0
    vote_reviews = 0
    started = time.perf_counter()

    try:
        iterator = iter_jsonl_gz(REVIEW_FILE, stats=stats)
        total = total_lines if limit is None else min(limit, total_lines)
        for record in tqdm(iterator, total=total, desc="解析评论", unit="条"):
            if limit is not None and rows >= limit:
                break
            overall = record.get("overall")
            asin = record.get("asin")
            if overall is not None:
                rating_min = overall if rating_min is None else min(rating_min, overall)
                rating_max = overall if rating_max is None else max(rating_max, overall)
            if asin is None:
                null_asin += 1

            style = _style_text(record.get("style"))
            vote = _vote_count(record.get("vote"))
            images = record.get("image")
            image_count = len(images) if isinstance(images, list) else 0
            if style:
                style_reviews += 1
            if vote is not None:
                vote_reviews += 1
            if image_count:
                image_reviews += 1

            columns["reviewerID"].append(_str(record.get("reviewerID")))
            columns["asin"].append(_str(asin))
            columns["reviewerName"].append(_str(record.get("reviewerName")))
            columns["overall"].append(overall)
            columns["verified"].append(record.get("verified"))
            columns["reviewTime"].append(_str(record.get("reviewTime")))
            columns["unixReviewTime"].append(record.get("unixReviewTime"))
            columns["reviewText"].append(_str(record.get("reviewText")))
            columns["summary"].append(_str(record.get("summary")))
            columns["style"].append(style)
            columns["vote"].append(vote)
            columns["image_count"].append(image_count)
            rows += 1

            if rows % chunk_size == 0:
                _flush(writer, columns, REVIEW_SCHEMA)
    finally:
        _flush(writer, columns, REVIEW_SCHEMA)
        writer.close()

    elapsed = time.perf_counter() - started
    assert rows > 0, "评论表为空"
    if limit is None:
        assert stats["bad"] == 0, f"评论中有 {stats['bad']} 行无法解析"
        assert rows == stats["lines"] - stats["empty"] - stats["bad"], "评论行数对不上"
    assert rating_min is None or rating_min >= 1.0, f"评分下界异常: {rating_min}"
    assert rating_max is None or rating_max <= 5.0, f"评分上界异常: {rating_max}"
    assert null_asin == 0, f"评论中有 {null_asin} 条缺少 asin"

    written = pq.ParquetFile(out_path).metadata.num_rows
    assert written == rows, f"落盘行数 {written} 与解析行数 {rows} 不一致"

    return {
        "rows": rows,
        "source_lines": stats["lines"],
        "empty_lines": stats["empty"],
        "bad_lines": stats["bad"],
        "rating_min": rating_min,
        "rating_max": rating_max,
        "null_asin": null_asin,
        "style_rows": style_reviews,
        "vote_rows": vote_reviews,
        "image_rows": image_reviews,
        "seconds": round(elapsed, 1),
        "output": str(out_path),
        "size_mb": round(out_path.stat().st_size / 1024**2, 1),
    }


def parse_meta(
    out_path: Path,
    total_lines: int,
    chunk_size: int,
    force: bool,
) -> dict:
    if out_path.exists() and not force:
        raise SystemExit(f"输出已存在，加 --force 覆盖: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"lines": 0, "empty": 0, "bad": 0}
    columns = _new_columns(META_SCHEMA)
    writer = pq.ParquetWriter(out_path, META_SCHEMA, compression="zstd")
    seen: set[str] = set()
    rows = 0
    duplicates = 0
    null_asin = 0
    price_parsed = 0
    also_buy_nonempty = 0
    started = time.perf_counter()

    try:
        for record in tqdm(
            iter_jsonl_gz(META_FILE, stats=stats),
            total=total_lines,
            desc="解析商品",
            unit="条",
        ):
            asin = _str(record.get("asin"))
            if asin is None:
                null_asin += 1
                continue
            if asin in seen:
                duplicates += 1
                continue
            seen.add(asin)

            raw_price = _str(record.get("price"))
            price = _parse_price(raw_price)
            if price is not None:
                price_parsed += 1

            also_buy = _str_list(record.get("also_buy"))
            if also_buy:
                also_buy_nonempty += 1

            description = record.get("description")
            if isinstance(description, list):
                description = "\n".join(str(item) for item in description).strip() or None
            else:
                description = _str(description)

            columns["asin"].append(asin)
            columns["title"].append(_str(record.get("title")))
            columns["brand"].append(_str(record.get("brand")))
            columns["price_raw"].append(raw_price[:200] if raw_price else None)
            columns["price"].append(price)
            columns["main_cat"].append(_str(record.get("main_cat")))
            columns["category"].append(_str_list(record.get("category")))
            columns["also_buy"].append(also_buy)
            columns["also_view"].append(_str_list(record.get("also_view")))
            columns["feature"].append(_str_list(record.get("feature")))
            columns["description"].append(description)
            columns["rank"].append(_str_list(record.get("rank")))
            columns["imageURL"].append(_str_list(record.get("imageURL")))
            columns["imageURLHighRes"].append(_str_list(record.get("imageURLHighRes")))
            columns["tech1"].append(_str(record.get("tech1")))
            columns["tech2"].append(_str(record.get("tech2")))
            columns["fit"].append(_str(record.get("fit")))
            columns["similar_item"].append(_str(record.get("similar_item")))
            columns["date"].append(_str(record.get("date")))
            rows += 1

            if rows % chunk_size == 0:
                _flush(writer, columns, META_SCHEMA)
    finally:
        _flush(writer, columns, META_SCHEMA)
        writer.close()

    elapsed = time.perf_counter() - started
    assert rows > 0, "商品表为空"
    assert stats["bad"] == 0, f"商品中有 {stats['bad']} 行无法解析"
    assert rows + duplicates + null_asin == total_lines, "商品行数对不上"
    assert null_asin == 0, f"商品中有 {null_asin} 条缺少 asin"
    assert price_parsed > 0, "没有解析出任何价格"

    written = pq.ParquetFile(out_path).metadata.num_rows
    assert written == rows, f"落盘行数 {written} 与去重后行数 {rows} 不一致"

    return {
        "rows": rows,
        "source_lines": stats["lines"],
        "duplicate_asins": duplicates,
        "null_asin": null_asin,
        "price_parsed": price_parsed,
        "also_buy_nonempty": also_buy_nonempty,
        "seconds": round(elapsed, 1),
        "output": str(out_path),
        "size_mb": round(out_path.stat().st_size / 1024**2, 1),
    }


def _total_ram_gb() -> float | None:
    if os.name != "nt":
        return None

    class MemoryStatusEx(ctypes.Structure):
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

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    try:
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except OSError:
        return None
    return round(status.ullTotalPhys / 1024**3, 1)


def _hardware() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _total_ram_gb(),
        "python": platform.python_version(),
        "pyarrow": pa.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="把原始 gzip JSONL 解析为 parquet")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="每批写入的行数")
    parser.add_argument("--limit", type=int, default=None, help="只解析评论的前 N 条，用于冒烟")
    parser.add_argument("--force", action="store_true", help="覆盖已有输出")
    parser.add_argument("--skip-reviews", action="store_true")
    parser.add_argument("--skip-meta", action="store_true")
    args = parser.parse_args()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)
    assert args.chunk_size > 0, "chunk-size 必须为正"

    started = time.perf_counter()
    log: dict = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "hardware": _hardware(),
        "args": vars(args),
    }

    if not args.skip_reviews:
        total = count_lines_gz(REVIEW_FILE)
        print(f"评论文件总行数 {total:,}")
        log["reviews"] = parse_reviews(
            REVIEWS_OUT, total, args.chunk_size, args.limit, args.force
        )

    if not args.skip_meta:
        total = count_lines_gz(META_FILE)
        print(f"商品文件总行数 {total:,}")
        log["meta"] = parse_meta(META_OUT, total, args.chunk_size, args.force)

    log["total_seconds"] = round(time.perf_counter() - started, 1)
    LOG_OUT.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成，用时 {log['total_seconds']} 秒，日志 {LOG_OUT}")


if __name__ == "__main__":
    main()
