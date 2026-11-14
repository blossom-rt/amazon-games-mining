"""生成静态业务看板：把四簇画像、核心指标与图表串成一个 HTML。

不依赖前端服务，直接打开 results/dashboard.html 即可。
用法：
    python src/build_dashboard.py
"""

from __future__ import annotations

import html
import json
from datetime import datetime

try:
    from .config import FIGURES, RESULTS, TABLES
except ImportError:
    from config import FIGURES, RESULTS, TABLES

OUT = RESULTS / "dashboard.html"
LABELS = {
    0: "中等热度 · 评分偏低",
    1: "高热度 · 共购密集",
    2: "长尾 · 共购稀疏",
    3: "中等热度 · Software",
}
ACTIONS = {
    0: "质量/退货排查：评分 3.64、verified 0.70，优先看差评主题。",
    1: "系列捆绑与交叉推荐：评论 94.5 条、also_buy 47.5，适合做组合位。",
    2: "搜索/自动化运营：45.8% 长尾、平均 5.9 条评论，不做人工精运营。",
    3: "价格带分析：价格观测率 0.393，适合做 Software 价格实验。",
}


def load(name: str) -> dict:
    path = TABLES / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def card(title: str, value: str, note: str) -> str:
    return (
        f'<div class="card"><div class="k">{html.escape(title)}</div>'
        f'<div class="v">{html.escape(value)}</div>'
        f'<div class="n">{html.escape(note)}</div></div>'
    )


def main() -> None:
    outcomes = load("cluster_outcomes.json")
    ablation = load("classify_ablation.json")
    terms = load("classify_terms.json")
    time_result = load("classify_time.json")
    networks = load("network_compare.json")
    validation = load("cluster_validation.json")

    clusters = outcomes.get("clusters", [])
    models = ablation.get("models", {})
    text_f1 = models.get("text_only", {}).get("macro_f1", {}).get("mean")
    combined_f1 = models.get("combined", {}).get("macro_f1", {}).get("mean")
    annual = time_result.get("annual", [])
    time_forward = next(
        (row for row in annual if row.get("year") == "all" and "2014" in row.get("direction", "")),
        {},
    )
    cross = {
        "also_buy": networks.get("also_buy", {}).get("cross_main_share"),
        "also_view": networks.get("also_view", {}).get("cross_main_share"),
        "co_reviewer": networks.get("co_reviewer", {}).get("cross_main_share"),
    }

    cluster_cards = []
    for row in clusters:
        cluster = int(row["cluster"])
        cluster_cards.append(
            f'<div class="cluster"><h3>簇 {cluster} · {html.escape(LABELS.get(cluster, ""))}</h3>'
            f'<p class="big">{row["n_products"]:,} 件</p>'
            f'<p>平均评分 {row["mean_overall"]:.3f} · 评论 {row["n_reviews"]:.1f} 条</p>'
            f'<p>verified {row["verified_share"]:.3f} · 价格观测 {row["price_observed"]:.3f}</p>'
            f'<p class="action">{html.escape(ACTIONS.get(cluster, ""))}</p></div>'
        )

    validation_rows = validation.get("outcomes", [])
    validation_html = "".join(
        f'<tr><td>{row["cluster"]}</td><td>{row["n_products"]:,}</td>'
        f'<td>{row["mean_overall"]:.3f}</td><td>{row["mean_n_reviews"]:.1f}</td>'
        f'<td>{row["verified_share"]:.3f}</td><td>{row["price_observed"]:.3f}</td></tr>'
        for row in validation_rows
    )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Amazon 游戏商品数据挖掘 · 业务看板</title>
<style>
body {{ font-family: "Microsoft YaHei", "PingFang SC", sans-serif; margin: 0; background: #f5f7fb; color: #1f2937; }}
header {{ background: #1f3a5f; color: white; padding: 26px 34px; }}
header h1 {{ margin: 0 0 6px; font-size: 24px; }}
header p {{ margin: 0; color: #cbd5e1; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px 20px 48px; }}
section {{ margin-top: 28px; }}
h2 {{ font-size: 18px; border-left: 4px solid #2563eb; padding-left: 10px; }}
.cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }}
.card, .cluster {{ background: white; border-radius: 10px; padding: 16px; box-shadow: 0 1px 3px rgba(15, 23, 42, .08); }}
.k {{ color: #64748b; font-size: 13px; }}
.v {{ font-size: 24px; font-weight: 700; color: #0f172a; margin: 6px 0; }}
.n {{ color: #64748b; font-size: 12px; line-height: 1.5; }}
.clusters {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }}
.cluster h3 {{ font-size: 14px; margin: 0 0 8px; color: #1d4ed8; }}
.big {{ font-size: 22px; font-weight: 700; margin: 4px 0; }}
.action {{ color: #475569; font-size: 12px; line-height: 1.6; }}
table {{ width: 100%; border-collapse: collapse; background: white; font-size: 13px; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #e2e8f0; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #eef2ff; color: #3730a3; }}
.note {{ color: #64748b; font-size: 12px; line-height: 1.7; }}
img {{ max-width: 100%; border-radius: 10px; background: white; padding: 8px; box-shadow: 0 1px 3px rgba(15, 23, 42, .08); }}
@media (max-width: 900px) {{
  .cards, .clusters {{ grid-template-columns: 1fr 1fr; }}
}}
</style>
</head>
<body>
<header>
  <h1>Amazon 游戏商品数据挖掘 · 业务看板</h1>
  <p>游戏本体样本 40,259 件商品 / 1,456,092 条评论 · 生成于 {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
</header>
<main>

<section>
  <h2>关键指标</h2>
  <div class="cards">
    {card("游戏本体商品", "40,259", "去重后 71,911 件中的 55.98%")}
    {card("建模评论", "1,456,092", "正类率 74.75%")}
    {card("仅文本宏 F1", f"{text_f1:.4f}" if text_f1 else "—", "文本+元数据 0.8635，元数据无稳定增量")}
    {card("时间外推宏 F1", f"{time_forward.get('macro_f1', 0):.4f}" if time_forward else "—", "≤2014 训练 → 2015+ 测试")}
  </div>
</section>

<section>
  <h2>四个运营分层</h2>
  <div class="clusters">{''.join(cluster_cards)}</div>
</section>

<section>
  <h2>留出集外部验证</h2>
  <table>
    <thead><tr><th>簇</th><th>商品数</th><th>平均评分</th><th>评论量</th><th>verified</th><th>价格观测率</th></tr></thead>
    <tbody>{validation_html}</tbody>
  </table>
  <p class="note">按 asin 70/30 切分，KMeans 只在训练集拟合；留出集按最近质心分配。评论量 ε²=0.437、also_buy ε²=0.479，评分 ε² 仅 0.006，说明簇主要按热度/共购结构分层，不是按评分分层。</p>
</section>

<section>
  <h2>文本与规则</h2>
  <p class="note">仅文本宏 F1 {text_f1:.4f}，合并元数据 {combined_f1:.4f}；否定范围特征把宏 F1 从 0.8689 提到 0.8785，否定子集错误率 15.95%→14.39%。跨主品类边占比：also_buy {cross['also_buy']:.4f}、also_view {cross['also_view']:.4f}、共同评论人 {cross['co_reviewer']:.4f}；also_buy 与共同评论人网络 Jaccard 仅 0.0102。</p>
</section>

<section>
  <h2>图表</h2>
  <p><img src="figures/fig42_cluster_outcomes.png" alt="簇业务结果"></p>
  <p><img src="figures/fig46_network_compare.png" alt="三张网络对比"></p>
</section>

<p class="note">说明：also_buy 是推荐位，不是真实订单；价格缺失 83.44%，价格只做事后画像；BGE 对照为小样本，不能外推全量。</p>
</main>
</body>
</html>
"""
    OUT.write_text(html_text, encoding="utf-8")
    print(f"写入 {OUT}")


if __name__ == "__main__":
    main()
