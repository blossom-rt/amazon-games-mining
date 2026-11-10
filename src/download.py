"""下载并校验数据集源文件。

默认只做校验。加 --force 才会重新下载。
数据文件不进版本控制，其他人克隆仓库后跑一次本脚本即可复现。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

try:
    from .config import REVIEW_URL, META_URL, REVIEW_FILE, META_FILE, SHA256
except ImportError:
    from config import REVIEW_URL, META_URL, REVIEW_FILE, META_FILE, SHA256


TARGETS = {
    REVIEW_FILE.name: REVIEW_URL,
    META_FILE.name: META_URL,
}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest().upper()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"下载 {url}")
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with tmp.open("wb") as out:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if total:
                    print(f"\r  {done / 2**20:7.1f} / {total / 2**20:7.1f} MB", end="")
    print()
    tmp.replace(dest)


def verify(path: Path, expected: str | None) -> bool:
    if not path.exists():
        print(f"[缺失] {path.name}")
        return False
    actual = sha256_of(path)
    if expected is None:
        print(f"[存在] {path.name}  {actual}")
        return True
    if actual != expected:
        print(f"[校验失败] {path.name}")
        print(f"  期望 {expected}")
        print(f"  实际 {actual}")
        return False
    print(f"[校验通过] {path.name}  {actual}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="下载或校验 Amazon Video_Games 数据文件")
    parser.add_argument("--verify", action="store_true", help="只校验，不下载")
    parser.add_argument("--force", action="store_true", help="即使文件已存在也重新下载")
    args = parser.parse_args()

    ok = True
    for name, url in TARGETS.items():
        dest = REVIEW_FILE.parent / name
        expected = SHA256.get(name)

        if args.force or (not args.verify and not dest.exists()):
            download(url, dest)

        if not verify(dest, expected):
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
