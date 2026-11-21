"""P6 一键复现入口：按顺序执行整条流水线并记录运行日志。

默认执行顺序：

    parse -> scope -> build_features -> governance -> rules -> classify -> cluster

`rules` 与 `cluster` 在 P3/P5 尚未实现时会被标记为 missing，不会中断流水线；
加 `--require-all` 可以把它们变成硬性要求。

用法：
    python src/run_all.py --check          # 只校验主流水线产物
    python src/run_all.py --check --extras # 连补充实验产物一起校验
    python src/run_all.py --dry-run        # 只打印将执行的命令
    python src/run_all.py                  # 主流水线全量重跑
    python src/run_all.py --extras         # 主流水线 + 补充实验 + 44 张图
    python src/run_all.py --only link_prediction
    python src/run_all.py --skip classify --keep-going
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

try:
    from .config import (
        DATA_PROCESSED,
        FIGURES,
        ROOT,
        TABLES,
        sanitize_result_jsons,
    )
except ImportError:
    from config import DATA_PROCESSED, FIGURES, ROOT, TABLES, sanitize_result_jsons

LOG_PATH = TABLES / "run_log.json"
SCHEMA_VERSION = 1

STEPS: list[dict] = [
    {
        "name": "parse",
        "script": "parse.py",
        "required": True,
        "force_arg": True,
        "outputs": [
            DATA_PROCESSED / "reviews.parquet",
            DATA_PROCESSED / "meta.parquet",
            TABLES / "parse_log.json",
        ],
        "figures": [],
    },
    {
        "name": "scope",
        "script": "scope.py",
        "required": True,
        "force_arg": True,
        "outputs": [
            DATA_PROCESSED / "scope.parquet",
            TABLES / "scope_summary.json",
        ],
        "figures": [],
    },
    {
        "name": "build_features",
        "script": "build_features.py",
        "required": True,
        "force_arg": True,
        "outputs": [
            DATA_PROCESSED / "features.parquet",
            DATA_PROCESSED / "folds.parquet",
            TABLES / "feature_summary.json",
        ],
        "figures": [],
    },
    {
        "name": "governance",
        "script": "governance.py",
        "required": True,
        "force_arg": True,
        "outputs": [
            TABLES / "governance_report.json",
            ROOT / "docs" / "governance_report.md",
        ],
        "figures": [],
    },
    {
        "name": "rules",
        "script": "rules.py",
        "required": False,
        "force_arg": True,
        "outputs": [
            TABLES / "rules_summary.json",
            TABLES / "rules_top.csv",
            TABLES / "rules_thresholds.csv",
            TABLES / "rules_runtime.csv",
        ],
        "figures": ["fig1[1-9]_*.png"],
        "min_figures": 5,
    },
    {
        "name": "classify",
        "script": "classify.py",
        "required": True,
        "force_arg": False,
        "outputs": [
            TABLES / "classify_metrics.csv",
            TABLES / "classify_summary.json",
            TABLES / "classify_errors.csv",
            TABLES / "classify_importance.csv",
            TABLES / "classify_shap.json",
            FIGURES / "fig30_classify_roc.png",
            FIGURES / "fig31_classify_pr.png",
            FIGURES / "fig32_classify_confusion.png",
            FIGURES / "fig33_classify_importance.png",
            FIGURES / "fig34_classify_shap.png",
            FIGURES / "fig35_classify_metrics.png",
        ],
        "figures": [],
    },
    {
        "name": "cluster",
        "script": "cluster.py",
        "required": False,
        "force_arg": True,
        "outputs": [
            TABLES / "cluster_summary.json",
            TABLES / "cluster_profiles.csv",
            TABLES / "cluster_stability.csv",
        ],
        "figures": ["fig2[0-9]_*.png"],
        "min_figures": 5,
    },
]

# 补充实验：默认不执行，`--extras` 时追加到主流水线之后。
EXTRAS: list[dict] = [
    {
        "name": "classify_decision",
        "script": "classify_decision.py",
        "required": True,
        "force_arg": False,
        "args": ["--limit", "200000"],
        "outputs": [
            TABLES / "classify_decision.json",
            TABLES / "classify_decision_curve.csv",
            TABLES / "classify_decision_oof.parquet",
            FIGURES / "fig48_decision_curve.png",
        ],
        "figures": [],
    },
    {
        "name": "classify_subgroups",
        "script": "classify_subgroups.py",
        "required": True,
        "force_arg": False,
        "outputs": [
            TABLES / "classify_subgroups.csv",
            TABLES / "classify_subgroups.json",
            FIGURES / "fig49_subgroup_errors.png",
        ],
        "figures": [],
    },
    {
        "name": "classify_tuning",
        "script": "classify_tuning.py",
        "required": True,
        "force_arg": False,
        "args": ["--limit", "80000"],
        "outputs": [
            TABLES / "classify_tuning.json",
            TABLES / "classify_tuning_metrics.csv",
        ],
        "figures": [],
    },
    {
        "name": "classify_imbalance",
        "script": "classify_imbalance.py",
        "required": True,
        "force_arg": False,
        "args": ["--limit", "200000"],
        "outputs": [
            TABLES / "classify_imbalance.csv",
            TABLES / "classify_imbalance.json",
        ],
        "figures": [],
    },
    {
        "name": "feature_selection",
        "script": "feature_selection.py",
        "required": True,
        "force_arg": False,
        "args": ["--limit", "50000"],
        "outputs": [
            TABLES / "feature_selection.csv",
            TABLES / "feature_selection.json",
            FIGURES / "fig50_feature_selection.png",
        ],
        "figures": [],
    },
    {
        "name": "classify_time",
        "script": "classify_time.py",
        "required": True,
        "force_arg": False,
        "args": ["--limit", "300000"],
        "outputs": [
            TABLES / "classify_time.json",
            TABLES / "classify_time_metrics.csv",
            FIGURES / "fig43_time_drift.png",
        ],
        "figures": [],
    },
    {
        "name": "link_prediction",
        "script": "link_prediction.py",
        "required": True,
        "force_arg": False,
        "args": ["--sources", "2000"],
        "outputs": [
            TABLES / "link_prediction.json",
            TABLES / "link_prediction_metrics.csv",
            FIGURES / "fig47_link_prediction.png",
        ],
        "figures": [],
    },
    {
        "name": "figures",
        "script": "figure_pipeline.py",
        "required": True,
        "force_arg": False,
        "args": ["--all"],
        "outputs": [],
        "figures": ["fig[0-9][0-9]_*.png"],
        "min_figures": 44,
    },
]

PACKAGES = [
    "pandas",
    "numpy",
    "pyarrow",
    "scipy",
    "scikit-learn",
    "mlxtend",
    "matplotlib",
    "seaborn",
    "shap",
    "lime",
]


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _total_ram_gb() -> float | None:
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
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

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
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
        "machine": platform.machine(),
    }


def _versions() -> dict:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_info() -> dict:
    def _run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    status = _run("status", "--porcelain")
    return {
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD") or None,
        "commit": _run("rev-parse", "HEAD") or None,
        "dirty": bool(status),
        "dirty_files": len(status.splitlines()),
    }


def _rows(path: Path) -> int | None:
    try:
        if path.suffix == ".parquet":
            return pq.ParquetFile(path).metadata.num_rows
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("rows"), int):
                return data["rows"]
            return None
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", errors="replace") as handle:
                return max(sum(1 for _ in handle) - 1, 0)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return None


def _artifact(path: Path) -> dict:
    exists = path.exists()
    return {
        "path": _relative(path),
        "exists": exists,
        "rows": _rows(path) if exists else None,
        "size_mb": round(path.stat().st_size / 1024**2, 2) if exists else None,
    }


def _figure_paths(step: dict) -> list[Path]:
    paths: list[Path] = []
    for pattern in step.get("figures", []):
        paths.extend(sorted(FIGURES.glob(pattern)))
    return paths


def _missing(step: dict) -> tuple[list[Path], list[str]]:
    missing = [path for path in step["outputs"] if not path.exists()]
    missing_figures: list[str] = []
    min_figures = step.get("min_figures", 0)
    if min_figures:
        found = len(set(_figure_paths(step)))
        if found < min_figures:
            missing_figures.append(f"figures={found} (<{min_figures})")
    return missing, missing_figures


def _command(step: dict, force: bool) -> list[str]:
    command = [sys.executable, str(ROOT / "src" / step["script"])]
    command.extend(step.get("args", []))
    if force and step.get("force_arg", True):
        command.append("--force")
    return command


def _run_step(step: dict, force: bool) -> dict:
    script = ROOT / "src" / step["script"]
    record: dict = {
        "name": step["name"],
        "required": step["required"],
        "script": _relative(script),
        "started_at": _now(),
    }

    if not script.exists():
        record.update(
            {
                "status": "missing",
                "command": None,
                "finished_at": _now(),
                "seconds": 0.0,
                "exit_code": None,
                "outputs": [_artifact(path) for path in step["outputs"]],
                "figures": [],
            }
        )
        return record

    command = _command(step, force)
    record["command"] = subprocess.list2cmdline(command)
    print(f"\n执行 {step['name']}：{record['command']}", flush=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    seconds = time.perf_counter() - started

    figures = _figure_paths(step)
    record.update(
        {
            "finished_at": _now(),
            "seconds": round(seconds, 1),
            "exit_code": result.returncode,
            "outputs": [_artifact(path) for path in step["outputs"]],
            "figures": [_relative(path) for path in figures],
            "status": "ok" if result.returncode == 0 else "failed",
        }
    )
    return record


def _check(steps: list[dict], require_all: bool) -> int:
    failures = 0
    print(f"{'步骤':<24}{'状态':<10}缺失产物")
    for step in steps:
        script = ROOT / "src" / step["script"]
        missing, missing_figures = _missing(step)
        if not script.exists():
            status = "missing" if not step["required"] else "error"
            detail = "脚本不存在"
        elif missing or missing_figures:
            status = "incomplete"
            detail = ", ".join([_relative(path) for path in missing] + missing_figures)
        else:
            status = "ok"
            detail = ""

        if status == "ok":
            pass
        elif step["required"] or require_all:
            failures += 1
        print(f"{step['name']:<24}{status:<10}{detail}")

    return 1 if failures else 0


def _dry_run(steps: list[dict], force: bool) -> None:
    for step in steps:
        script = ROOT / "src" / step["script"]
        if not script.exists():
            print(f"[dry-run] {step['name']}: 跳过，脚本 {_relative(script)} 不存在")
            continue
        print(f"[dry-run] {step['name']}: {subprocess.list2cmdline(_command(step, force))}")


def _selected(only: set[str], skip: set[str], extras: bool) -> list[dict]:
    all_steps = STEPS + EXTRAS
    if only:
        pool = all_steps
    else:
        pool = STEPS + (EXTRAS if extras else [])
    selected = [step for step in pool if (not only or step["name"] in only) and step["name"] not in skip]
    unknown = (only | skip) - {step["name"] for step in all_steps}
    if unknown:
        raise SystemExit(f"未知步骤：{', '.join(sorted(unknown))}")
    return selected


def _write_log(payload: dict) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="一键执行数据挖掘流水线并记录运行日志")
    parser.add_argument("--only", default="", help="只跑这些步骤，逗号分隔")
    parser.add_argument("--skip", default="", help="跳过这些步骤，逗号分隔")
    parser.add_argument("--check", action="store_true", help="只校验现有产物，不执行")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")
    parser.add_argument("--no-force", action="store_true", help="不向脚本传 --force")
    parser.add_argument("--keep-going", action="store_true", help="某一步失败后继续")
    parser.add_argument("--require-all", action="store_true", help="rules/cluster 缺失也视为失败")
    parser.add_argument("--extras", action="store_true", help="在主流水线后追加补充实验")
    parser.add_argument("--list", action="store_true", help="列出步骤")
    args = parser.parse_args()

    if args.list:
        for group, steps in (("主流水线", STEPS), ("补充实验", EXTRAS)):
            print(f"\n{group}")
            for step in steps:
                flag = "必需" if step["required"] else "可选"
                print(f"  {step['name']:<20}{flag:<6}{step['script']}")
        return

    only = {item.strip() for item in args.only.split(",") if item.strip()}
    skip = {item.strip() for item in args.skip.split(",") if item.strip()}

    if args.check:
        check_steps = _selected(only, skip, args.extras)
        raise SystemExit(_check(check_steps, args.require_all))

    selected = _selected(only, skip, args.extras)
    force = not args.no_force

    if args.dry_run:
        _dry_run(selected, force)
        return

    git_info = _git_info()
    hardware = _hardware()
    packages = _versions()
    started = time.perf_counter()
    records: list[dict] = []
    failed = False

    for step in selected:
        record = _run_step(step, force)
        records.append(record)

        if record["status"] == "failed":
            failed = True
            if not args.keep_going:
                break
        elif record["status"] == "missing" and args.require_all:
            failed = True
            if not args.keep_going:
                break

    total_seconds = round(time.perf_counter() - started, 1)
    status = "failed" if failed else "ok"
    if not failed and any(record["status"] == "missing" for record in records):
        status = "partial"

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "git": git_info,
        "hardware": hardware,
        "packages": packages,
        "args": {
            "only": sorted(only),
            "skip": sorted(skip),
            "force": force,
            "keep_going": args.keep_going,
            "require_all": args.require_all,
            "extras": args.extras,
        },
        "steps": records,
        "total_seconds": total_seconds,
        "status": status,
    }
    _write_log(payload)

    print(f"\n总耗时 {total_seconds} 秒，状态 {status}")
    for record in records:
        print(f"  {record['name']:<16}{record['status']:<10}{record.get('seconds', 0):>8.1f} 秒")
    print(f"运行日志 {_relative(LOG_PATH)}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
