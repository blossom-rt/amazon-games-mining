"""重绘并校验仓库中的 44 张结果图。

权威入口分两类输入：
1. 已提交的 results/tables 与 data/processed 中间表，供重绘脚本使用；
2. 由 classify/cluster/补充实验脚本直接产出的少量图（见 EXPERIMENT_ONLY）。

用法：
    python src/figure_pipeline.py --check
    python src/figure_pipeline.py --all
    python src/figure_pipeline.py --only eda,classify
    python src/figure_pipeline.py --list
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from PIL import Image

from config import sanitize_result_jsons

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"
MIN_WIDTH = 600
MIN_HEIGHT = 300

EXPECTED: dict[str, list[str]] = {
    "eda": [
        "fig01_rating_distribution.png",
        "fig02_class_balance.png",
        "fig03_reviews_over_time.png",
        "fig04_main_cat.png",
        "fig05_basket_length.png",
        "fig06_price_distribution.png",
        "fig07_review_length.png",
        "fig08_missing_rates.png",
        "fig09_reviews_per_product.png",
        "fig10_vote_style_by_rating.png",
    ],
    "rules": [
        "fig11_basket_length.png",
        "fig12_rule_scatter.png",
        "fig13_top_lift.png",
        "fig14_runtime.png",
        "fig15_lift_distribution.png",
        "fig16_top_items.png",
    ],
    "cluster": [
        "fig20_elbow.png",
        "fig21_silhouette.png",
        "fig22_clusters_pca.png",
        "fig23_profile_heatmap.png",
        "fig24_cluster_sizes.png",
        "fig25_bootstrap_ari.png",
        "fig26_price_reviews.png",
        "fig39_cluster_methods.png",
        "fig45_cluster_validation.png",
    ],
    "classify": [
        "fig30_classify_roc.png",
        "fig31_classify_pr.png",
        "fig32_classify_confusion.png",
        "fig33_classify_importance.png",
        "fig34_classify_shap.png",
        "fig35_classify_metrics.png",
        "fig36_calibration.png",
        "fig37_threshold.png",
        "fig38_embedding_compare.png",
    ],
    "insights": [
        "fig40_ablation.png",
        "fig41_top_terms.png",
        "fig42_cluster_outcomes.png",
        "fig43_time_drift.png",
        "fig44_negation_scope.png",
        "fig46_network_compare.png",
        "fig47_link_prediction.png",
        "fig48_decision_curve.png",
        "fig49_subgroup_errors.png",
        "fig50_feature_selection.png",
    ],
}
ALL_FIGURES = [name for group in EXPECTED.values() for name in group]
if len(ALL_FIGURES) != 44 or len(set(ALL_FIGURES)) != 44:
    raise RuntimeError("EXPECTED 必须恰好包含 44 个不重复文件名")

# 这些图由实验脚本直接写成最终样式，不从 tables 重绘：
# classify.py / cluster.py / classify_time.py / classify_subgroups.py / feature_selection.py
EXPERIMENT_ONLY = {
    "fig20_elbow.png",
    "fig21_silhouette.png",
    "fig23_profile_heatmap.png",
    "fig25_bootstrap_ari.png",
    "fig30_classify_roc.png",
    "fig31_classify_pr.png",
    "fig32_classify_confusion.png",
    "fig33_classify_importance.png",
    "fig43_time_drift.png",
    "fig49_subgroup_errors.png",
    "fig50_feature_selection.png",
}

RENDERERS = {
    "report": "redo_report_figures",
    "eda": "redo_eda_rules_figs",
    "classify": "redo_classify_figs",
    "other": "restyle_other_figures",
}
RENDER_ORDER = ["report", "eda", "classify", "other"]
ORDER_HINT = (
    "缺少由实验脚本直接产出的图。请先在仓库根目录执行：\n"
    "  python src/run_all.py --extras\n"
    "若只缺 fig43，再执行：\n"
    "  python src/classify_time.py --limit 300000\n"
    "若只缺 fig49/fig50，再执行：\n"
    "  python src/classify_subgroups.py\n"
    "  python src/feature_selection.py --limit 50000"
)


def _font_warning() -> None:
    try:
        from matplotlib import font_manager
    except ImportError:
        return
    names = {font.name for font in font_manager.fontManager.ttflist}
    candidates = {"Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"}
    if not names & candidates:
        print(
            "警告：未找到中文字体（Microsoft YaHei / SimHei / Noto Sans CJK SC / PingFang SC）。"
            "图中可能显示方框，请按 docs/figure_reproduction.md 安装字体后重跑。",
            file=sys.stderr,
        )


def _image_ok(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "缺失"
    if path.stat().st_size == 0:
        return False, "空文件"
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except Exception as exc:  # noqa: BLE001 - 需要把具体坏图报给用户
        return False, f"无法读取：{exc}"
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return False, f"尺寸过小：{width}×{height}"
    return True, f"{width}×{height}"


def check() -> int:
    failures = 0
    print(f"目标图数：{len(ALL_FIGURES)}")
    for group, names in EXPECTED.items():
        bad = []
        for name in names:
            ok, detail = _image_ok(FIGURES / name)
            if not ok:
                bad.append(f"{name} ({detail})")
        status = "ok" if not bad else "FAIL"
        print(f"  {group:<10}{status:<6}{len(names) - len(bad)}/{len(names)}")
        for item in bad:
            print(f"    - {item}")
        failures += len(bad)

    extra = sorted(path.name for path in FIGURES.glob("fig*.png") if path.name not in ALL_FIGURES)
    if extra:
        print("提示：以下图不在 44 张清单内，未计入：")
        for name in extra:
            print(f"  - {name}")

    if failures:
        print(f"\n结果：{len(ALL_FIGURES) - failures}/{len(ALL_FIGURES)}，缺少或损坏 {failures} 张。")
        missing = {path.name for path in FIGURES.glob("fig*.png")}
        if (set(ALL_FIGURES) - missing) & EXPERIMENT_ONLY:
            print(ORDER_HINT)
        return 1
    print(f"\n结果：{len(ALL_FIGURES)}/{len(ALL_FIGURES)}，全部可读。")
    return 0


def render(only: list[str]) -> int:
    unknown = set(only) - set(RENDERERS)
    if unknown:
        raise SystemExit(f"未知渲染模块：{', '.join(sorted(unknown))}；可选 {', '.join(RENDERERS)}")
    _font_warning()
    selected = [name for name in RENDER_ORDER if not only or name in only]
    for name in selected:
        module_name = RENDERERS[name]
        print(f"\n渲染模块 {name}：{module_name}")
        module = importlib.import_module(module_name)
        try:
            module.main()
        except FileNotFoundError as exc:
            print(f"缺少输入：{exc}", file=sys.stderr)
            print(ORDER_HINT, file=sys.stderr)
            return 2
    sanitize_result_jsons()
    return check()


def main() -> None:
    parser = argparse.ArgumentParser(description="重绘并校验 44 张结果图")
    parser.add_argument("--check", action="store_true", help="只检查现有图片，不重绘")
    parser.add_argument("--all", action="store_true", help="重绘所有可从 tables 复现的图并检查")
    parser.add_argument("--only", default="", help="只跑 report/eda/classify/other，逗号分隔")
    parser.add_argument("--list", action="store_true", help="列出图与渲染模块")
    args = parser.parse_args()

    if args.list:
        for group, names in EXPECTED.items():
            print(f"\n{group}")
            for name in names:
                flag = "实验脚本直出" if name in EXPERIMENT_ONLY else "tables 重绘"
                print(f"  {name:<36}{flag}")
        return
    if args.check:
        raise SystemExit(check())
    only = [item.strip() for item in args.only.split(",") if item.strip()]
    if not args.all and not only:
        parser.error("请指定 --check、--all 或 --only")
    raise SystemExit(render(only))


if __name__ == "__main__":
    main()
