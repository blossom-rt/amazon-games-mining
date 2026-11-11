"""读取 Amazon 评论与元数据的工具函数。

文件格式为 gzip 压缩的 JSON Lines，一行一个对象。
这里全部采用流式读取，一次只持有一行，避免把整个文件读进内存。
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator, Any


def iter_jsonl_gz(
    path: str | Path,
    stats: dict[str, int] | None = None,
) -> Iterator[dict[str, Any]]:
    """逐行读取 gzip 压缩的 JSON Lines 文件。

    解析失败的行会被跳过，由调用方通过返回值感知数据质量。
    这里选择跳过而不是抛错，因为原始数据中确实存在少量无法解析的行。

    传入 stats 时会在原字典上累加 lines（总行数）、empty（空行）、
    bad（解析失败行），方便解析脚本核对记录数。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")

    if stats is not None:
        stats.setdefault("lines", 0)
        stats.setdefault("empty", 0)
        stats.setdefault("bad", 0)

    with gzip.open(path, "rb") as f:
        for line in f:
            if stats is not None:
                stats["lines"] += 1
            line = line.strip()
            if not line:
                if stats is not None:
                    stats["empty"] += 1
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if stats is not None:
                    stats["bad"] += 1
                continue


def count_lines_gz(path: str | Path) -> int:
    """统计 gzip 文件的行数，只解压不保留对象，用于核对官方给的记录数。"""
    path = Path(path)
    n = 0
    with gzip.open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def peek_jsonl_gz(path: str | Path, n: int = 3) -> list[dict[str, Any]]:
    """取前 n 条记录，用于快速查看结构，不读取整个文件。"""
    out: list[dict[str, Any]] = []
    for i, obj in enumerate(iter_jsonl_gz(path)):
        if i >= n:
            break
        out.append(obj)
    return out
