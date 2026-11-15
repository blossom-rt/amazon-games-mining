"""生成可离线打开、可部署到静态托管的交互看板。

数据来自 results/tables 下的 JSON，不重新训练模型。
输出 results/dashboard.html，可直接用浏览器打开，也可通过静态托管访问。

用法：
    python src/build_static_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .config import ROOT, TABLES
except ImportError:
    from config import ROOT, TABLES

OUT = ROOT / "results" / "dashboard.html"


def _load(name: str) -> dict:
    return json.loads((TABLES / name).read_text(encoding="utf-8"))


def main() -> None:
    clusters = _load("cluster_outcomes.json")["clusters"]
    time_data = _load("classify_time.json")
    network = _load("network_compare.json")
    summary = _load("classify_summary.json")

    annual = {"forward": [], "backward": []}
    for row in time_data["annual"]:
        if row["year"] == "all":
            continue
        key = "forward" if row["direction"].startswith("2014") else "backward"
        annual[key].append(
            {
                "year": int(row["year"]),
                "macro_f1": row["macro_f1"],
                "pr_auc": row["pr_auc"],
                "roc_auc": row["roc_auc"],
                "n": row["n"],
            }
        )
    for key in annual:
        annual[key].sort(key=lambda item: item["year"])

    payload = {
        "clusters": clusters,
        "annual": annual,
        "network": {
            "also_buy": network["also_buy"],
            "also_view": network["also_view"],
            "co_reviewer": network["co_reviewer"],
            "overlap": network["overlap"],
        },
        "kpi": {
            "products": 40259,
            "reviews": 1456092,
            "positive_rate": summary["positive_rate"],
            "text_f1": 0.8727,
        },
    }

    html = TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"写入 {OUT}")


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Amazon 游戏商品数据挖掘 · 交互看板</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root { --navy:#16324f; --blue:#2563eb; --line:#dbe4f0; --muted:#64748b; }
* { box-sizing: border-box; }
body { margin:0; font-family:"Microsoft YaHei","PingFang SC",sans-serif; background:#f4f7fb; color:#1f2937; }
header { background:linear-gradient(120deg,#16324f,#1e4f7a); color:#fff; padding:26px 34px; }
header h1 { margin:0 0 6px; font-size:24px; }
header p { margin:0; color:#cfe0f1; font-size:13px; }
.wrap { max-width:1240px; margin:0 auto; padding:20px 22px 46px; }
.filters { display:grid; grid-template-columns:2fr 1fr; gap:16px; background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:16px; }
.filters h3 { margin:0 0 8px; font-size:13px; color:var(--muted); font-weight:600; }
.checks { display:flex; flex-wrap:wrap; gap:12px; }
.checks label { font-size:13px; background:#eef4fc; padding:6px 9px; border-radius:6px; cursor:pointer; }
.checks input { margin-right:4px; }
.yearbox { display:flex; align-items:center; gap:10px; }
input[type=range] { width:100%; accent-color:var(--blue); }
.tabs { display:flex; gap:6px; margin:0 0 14px; flex-wrap:wrap; }
.tabs button { border:1px solid var(--line); background:#fff; color:#334155; padding:8px 15px; border-radius:8px; cursor:pointer; font-size:13px; }
.tabs button.active { background:var(--navy); border-color:var(--navy); color:#fff; }
.tab { display:none; }
.tab.active { display:block; }
.cards { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:14px; }
.card, .panel { background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
.card .k { color:var(--muted); font-size:12px; }
.card .v { font-size:24px; font-weight:700; margin:5px 0 2px; color:#0f172a; }
.card .n { color:var(--muted); font-size:11px; line-height:1.5; }
.panel h2 { margin:0 0 4px; font-size:16px; }
.panel p { margin:0 0 10px; color:var(--muted); font-size:12px; line-height:1.6; }
.grid2 { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:14px; }
.chart { width:100%; height:330px; }
table { width:100%; border-collapse:collapse; font-size:12px; }
th, td { border-bottom:1px solid #e8eef6; padding:7px 6px; text-align:right; }
th:first-child, td:first-child { text-align:left; }
th { color:var(--muted); font-weight:600; background:#f8fafc; }
.note { margin-top:12px; color:var(--muted); font-size:12px; line-height:1.7; }
@media (max-width:900px) { .filters,.grid2 { grid-template-columns:1fr; } .cards { grid-template-columns:1fr 1fr; } }
</style>
</head>
<body>
<header>
  <h1>Amazon 游戏商品数据挖掘 · 交互看板</h1>
  <p>游戏本体 40,259 件商品 / 1,456,092 条评论；选择运营簇与评论年份后，总览 KPI、分群图与时间曲线同步更新。</p>
</header>
<div class="wrap">
  <div class="filters">
    <div>
      <h3>运营簇（影响总览与分群视图）</h3>
      <div class="checks" id="clusterChecks"></div>
    </div>
    <div>
      <h3>评论年份（影响总览时间 F1 与时间视图）</h3>
      <div class="yearbox">
        <input id="yearRange" type="range" min="1999" max="2018" value="2018" step="1">
        <strong id="yearLabel">2018</strong>
      </div>
    </div>
  </div>

  <div class="tabs">
    <button class="active" data-tab="overview">总览</button>
    <button data-tab="clusters">分群</button>
    <button data-tab="time">时间</button>
    <button data-tab="network">网络</button>
  </div>

  <section id="overview" class="tab active">
    <div class="cards">
      <div class="card"><div class="k">筛选商品数</div><div class="v" id="kpiProducts">—</div><div class="n">按所选运营簇汇总</div></div>
      <div class="card"><div class="k">加权平均评分</div><div class="v" id="kpiRating">—</div><div class="n">按商品数加权</div></div>
      <div class="card"><div class="k">平均评论量</div><div class="v" id="kpiReviews">—</div><div class="n">每件商品的平均评论数</div></div>
      <div class="card"><div class="k">时间外推宏 F1</div><div class="v" id="kpiTime">—</div><div class="n" id="kpiTimeNote">所选年份</div></div>
    </div>
    <div class="panel">
      <h2>所选簇的商品规模</h2>
      <p>勾选运营簇后，图表与 KPI 同时更新。</p>
      <div id="clusterSizeChart" class="chart"></div>
    </div>
    <div class="note">表格与图表来自 cluster_outcomes.json、classify_time.json、network_compare.json；also_buy 是推荐位网络，不是真实订单。</div>
  </section>

  <section id="clusters" class="tab">
    <div class="grid2">
      <div class="panel"><h2>平均评分</h2><p>评分差异很弱，簇主要分开热度与共购结构。</p><div id="ratingChart" class="chart"></div></div>
      <div class="panel"><h2>平均评论量</h2><p>高热度簇与长尾簇的差距远大于评分差距。</p><div id="reviewsChart" class="chart"></div></div>
    </div>
    <div class="panel" style="margin-top:14px">
      <h2>簇画像</h2>
      <table><thead><tr><th>簇</th><th>商品数</th><th>平均评分</th><th>评论量</th><th>verified</th><th>价格观测率</th><th>also_buy</th></tr></thead><tbody id="clusterTable"></tbody></table>
    </div>
  </section>

  <section id="time" class="tab">
    <div class="panel">
      <h2>按年份的宏 F1</h2>
      <p>正向：2014 及以前训练、2015+ 测试；反向：2015+ 训练、2014 及以前测试。滑块控制显示到哪一年。</p>
      <div id="timeChart" class="chart"></div>
    </div>
  </section>

  <section id="network" class="tab">
    <div class="grid2">
      <div class="panel"><h2>跨主品类边占比</h2><p>also_buy 比 also_view、共同评论人更窄。</p><div id="crossChart" class="chart"></div></div>
      <div class="panel"><h2>边重叠（Jaccard）</h2><p>also_buy 与共同评论人网络几乎不重叠。</p><div id="jaccardChart" class="chart"></div></div>
    </div>
  </section>
</div>

<script>
const DATA = __DATA__;
const clusters = DATA.clusters;
const selected = new Set(clusters.map(c => c.cluster));
let selectedYear = Number(document.getElementById('yearRange').value);

const clusterNames = {
  0: "簇 0 · 中等热度/偏低评分",
  1: "簇 1 · 高热度/共购密集",
  2: "簇 2 · 长尾/共购稀疏",
  3: "簇 3 · 中等热度/Software"
};

const checks = document.getElementById('clusterChecks');
clusters.forEach(c => {
  const label = document.createElement('label');
  label.innerHTML = `<input type="checkbox" checked value="${c.cluster}"> ${clusterNames[c.cluster]}`;
  label.querySelector('input').addEventListener('change', event => {
    if (event.target.checked) selected.add(c.cluster); else selected.delete(c.cluster);
    updateAll();
  });
  checks.appendChild(label);
});

document.querySelectorAll('.tabs button').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.tabs button').forEach(item => item.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(item => item.classList.remove('active'));
    button.classList.add('active');
    document.getElementById(button.dataset.tab).classList.add('active');
    window.dispatchEvent(new Event('resize'));
  });
});

document.getElementById('yearRange').addEventListener('input', event => {
  selectedYear = Number(event.target.value);
  document.getElementById('yearLabel').textContent = selectedYear;
  updateAll();
});

function rows() {
  return clusters.filter(c => selected.has(c.cluster));
}

function annualRow() {
  const group = selectedYear >= 2015 ? DATA.annual.forward : DATA.annual.backward;
  return group.find(item => item.year === selectedYear) || group[group.length - 1];
}

function updateKpis() {
  const chosen = rows();
  const products = chosen.reduce((sum, c) => sum + c.n_products, 0);
  const weightedRating = products ? chosen.reduce((sum, c) => sum + c.mean_overall * c.n_products, 0) / products : 0;
  const weightedReviews = products ? chosen.reduce((sum, c) => sum + c.n_reviews * c.n_products, 0) / products : 0;
  const row = annualRow();
  document.getElementById('kpiProducts').textContent = products.toLocaleString();
  document.getElementById('kpiRating').textContent = weightedRating.toFixed(3);
  document.getElementById('kpiReviews').textContent = weightedReviews.toFixed(1);
  document.getElementById('kpiTime').textContent = row ? row.macro_f1.toFixed(4) : '—';
  document.getElementById('kpiTimeNote').textContent = row
    ? `${selectedYear} 年测试宏 F1（n=${row.n.toLocaleString()}）`
    : '该年份无留出样本';
}

function updateClusterCharts() {
  const chosen = rows();
  Plotly.react('clusterSizeChart', [{
    x: chosen.map(c => 'C' + c.cluster),
    y: chosen.map(c => c.n_products),
    type: 'bar',
    marker: {color: '#2563eb'},
    text: chosen.map(c => c.n_products.toLocaleString()),
    textposition: 'auto'
  }], {margin:{t:20,r:10,b:70,l:60}, yaxis:{title:'商品数'}, xaxis:{tickangle:-12}}, {responsive:true});

  Plotly.react('ratingChart', [{
    x: chosen.map(c => 'C' + c.cluster), y: chosen.map(c => c.mean_overall),
    type: 'bar', marker: {color: '#0f766e'}, text: chosen.map(c => c.mean_overall.toFixed(3)), textposition: 'auto'
  }], {margin:{t:20,r:10,b:70,l:50}, yaxis:{title:'平均评分', range:[3.4,4.2]}, xaxis:{tickangle:-12}}, {responsive:true});

  Plotly.react('reviewsChart', [{
    x: chosen.map(c => 'C' + c.cluster), y: chosen.map(c => c.n_reviews),
    type: 'bar', marker: {color: '#d97706'}, text: chosen.map(c => c.n_reviews.toFixed(1)), textposition: 'auto'
  }], {margin:{t:20,r:10,b:70,l:50}, yaxis:{title:'平均评论量'}, xaxis:{tickangle:-12}}, {responsive:true});

  document.getElementById('clusterTable').innerHTML = chosen.map(c => `<tr>
    <td>${clusterNames[c.cluster]}</td><td>${c.n_products.toLocaleString()}</td>
    <td>${c.mean_overall.toFixed(3)}</td><td>${c.n_reviews.toFixed(1)}</td>
    <td>${c.verified_share.toFixed(3)}</td><td>${c.price_observed.toFixed(4)}</td><td>${c.mean_also_buy.toFixed(1)}</td>
  </tr>`).join('');
}

function updateTimeChart() {
  const f = DATA.annual.forward.filter(item => item.year <= selectedYear);
  const b = DATA.annual.backward.filter(item => item.year <= selectedYear);
  Plotly.react('timeChart', [
    {x:f.map(d=>d.year), y:f.map(d=>d.macro_f1), mode:'lines+markers', name:'≤2014 → 2015+', line:{color:'#2563eb'}},
    {x:b.map(d=>d.year), y:b.map(d=>d.macro_f1), mode:'lines+markers', name:'2015+ → ≤2014', line:{color:'#dc2626'}}
  ], {margin:{t:20,r:10,b:50,l:60}, yaxis:{title:'宏 F1'}, xaxis:{title:'评论年份'}, legend:{orientation:'h'}}, {responsive:true});
}

function staticCharts() {
  const n = DATA.network;
  const labels = ['also_buy', 'also_view', '共同评论人'];
  const values = [n.also_buy.cross_main_share, n.also_view.cross_main_share, n.co_reviewer.cross_main_share];
  Plotly.newPlot('crossChart', [{x:labels, y:values, type:'bar', marker:{color:['#2563eb','#d97706','#0f766e']}, text:values.map(v=>v.toFixed(4)), textposition:'auto'}],
    {margin:{t:20,r:10,b:40,l:60}, yaxis:{title:'跨主品类边占比'}}, {responsive:true});
  const j = [n.overlap.buy_view_jaccard, n.overlap.buy_co_jaccard, n.overlap.view_co_jaccard];
  Plotly.newPlot('jaccardChart', [{x:['buy ∩ view','buy ∩ co','view ∩ co'], y:j, type:'bar', marker:{color:'#7c3aed'}, text:j.map(v=>v.toFixed(4)), textposition:'auto'}],
    {margin:{t:20,r:10,b:40,l:60}, yaxis:{title:'Jaccard'}}, {responsive:true});
}

function updateAll() {
  updateKpis();
  updateClusterCharts();
  updateTimeChart();
}

staticCharts();
updateAll();
const initialTab = new URLSearchParams(location.search).get('tab');
if (initialTab) {
  const button = document.querySelector(`.tabs button[data-tab="${initialTab}"]`);
  if (button) button.click();
}
window.addEventListener('load', () => {
  ['clusterSizeChart','ratingChart','reviewsChart','timeChart','crossChart','jaccardChart']
    .forEach(id => { const el = document.getElementById(id); if (el) Plotly.Plots.resize(el); });
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
