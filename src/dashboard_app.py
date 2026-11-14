"""交互式业务看板：Streamlit + Plotly。

运行：
    streamlit run src/dashboard_app.py --server.port 8501

看板包含四个视图：总览、分群、时间、网络。
侧边栏的“运营簇”和“评论年份”两个筛选器会同时驱动多个视图：
    - 运营簇：总览 KPI、分群散点/柱状/表格；
    - 评论年份：总览的时间外推 KPI、时间曲线的年份范围与表格。
数据来自 results/tables/ 下的 JSON/CSV，不读取原始 parquet。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"

CLUSTER_LABELS = {
    0: "中等热度 · 评分偏低",
    1: "高热度 · 共购密集",
    2: "长尾 · 共购稀疏",
    3: "中等热度 · Software",
}
CLUSTER_COLORS = {
    "簇 0": "#5B8FF9",
    "簇 1": "#F6BD16",
    "簇 2": "#5AD8A6",
    "簇 3": "#E8684A",
}


def load_json(name: str) -> dict:
    path = TABLES / name
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_data() -> dict:
    clusters = pd.DataFrame(load_json("cluster_outcomes.json")["clusters"])
    clusters["cluster_label"] = clusters["cluster"].map(
        lambda value: f"簇 {value} · {CLUSTER_LABELS.get(int(value), '')}"
    )
    clusters["short_label"] = clusters["cluster"].map(lambda value: f"簇 {value}")

    annual = pd.DataFrame(load_json("classify_time.json")["annual"])
    annual = annual[annual["year"] != "all"].copy()
    annual["year"] = annual["year"].astype(int)
    annual["direction"] = annual["direction"].replace(
        {
            "2014及以前→2015+": "≤2014 → 2015+",
            "2015+→2014及以前": "2015+ → ≤2014",
        }
    )

    network = load_json("network_compare.json")
    network_rows = []
    for key, label in [
        ("also_buy", "also_buy"),
        ("also_view", "also_view"),
        ("co_reviewer", "共同评论人"),
    ]:
        item = network[key]
        network_rows.append(
            {
                "网络": label,
                "边数": item["edges"],
                "跨主品类边占比": item["cross_main_share"],
                "跨叶子边占比": item["cross_leaf_share"],
                "多边占比": item["multi_edge_share"],
                "边权中位数": item["median_edge_weight"],
            }
        )
    networks = pd.DataFrame(network_rows)
    overlap = pd.DataFrame(
        [
            {"对比": "also_buy ∩ also_view", "Jaccard": network["overlap"]["buy_view_jaccard"]},
            {"对比": "also_buy ∩ 共同评论人", "Jaccard": network["overlap"]["buy_co_jaccard"]},
            {"对比": "also_view ∩ 共同评论人", "Jaccard": network["overlap"]["view_co_jaccard"]},
        ]
    )

    ablation = load_json("classify_ablation.json")
    model_rows = []
    for key in ["text_only", "metadata_only", "combined", "dummy"]:
        item = ablation["models"][key]
        model_rows.append(
            {
                "模型": {
                    "text_only": "仅文本",
                    "metadata_only": "仅元数据",
                    "combined": "文本+元数据",
                    "dummy": "Dummy",
                }[key],
                "宏 F1": item["macro_f1"]["mean"],
                "PR-AUC": item["pr_auc"]["mean"],
                "ROC-AUC": item["roc_auc"]["mean"],
            }
        )
    return {
        "clusters": clusters,
        "annual": annual,
        "networks": networks,
        "overlap": overlap,
        "models": pd.DataFrame(model_rows),
    }


def metric_row(cluster_frame: pd.DataFrame, annual_frame: pd.DataFrame) -> dict:
    products = int(cluster_frame["n_products"].sum())
    if products == 0:
        return {"products": 0, "rating": 0.0, "reviews": 0.0, "price": 0.0, "time_f1": float("nan")}

    def weighted(column: str) -> float:
        return float((cluster_frame[column] * cluster_frame["n_products"]).sum() / products)

    forward = annual_frame[annual_frame["direction"] == "≤2014 → 2015+"]
    time_f1 = float(forward["macro_f1"].mean()) if len(forward) else float("nan")
    return {
        "products": products,
        "rating": weighted("mean_overall"),
        "reviews": weighted("n_reviews"),
        "price": weighted("price_observed"),
        "time_f1": time_f1,
    }


st.set_page_config(
    page_title="Amazon 游戏商品数据挖掘 · 交互看板",
    page_icon="🎮",
    layout="wide",
)

data = load_data()
clusters: pd.DataFrame = data["clusters"]
annual: pd.DataFrame = data["annual"]
networks: pd.DataFrame = data["networks"]
overlap: pd.DataFrame = data["overlap"]
models: pd.DataFrame = data["models"]

st.title("Amazon 游戏商品数据挖掘 · 交互看板")
st.caption(
    "游戏本体 40,259 件商品 / 1,456,092 条评论；筛选器会联动总览、分群与时间视图。"
)

with st.sidebar:
    st.header("筛选器")
    all_clusters = sorted(clusters["cluster"].unique().tolist())
    selected_clusters = st.multiselect(
        "运营簇（影响总览与分群）",
        options=all_clusters,
        default=all_clusters,
        format_func=lambda value: f"簇 {value} · {CLUSTER_LABELS.get(int(value), '')}",
    )
    min_year = int(annual["year"].min())
    max_year = int(annual["year"].max())
    year_range = st.slider(
        "评论年份（影响总览时间 KPI 与时间视图）",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
    )
    st.divider()
    st.caption(
        "联动说明：运营簇筛选同时改变总览 KPI 和分群视图；"
        "年份筛选同时改变总览时间 F1 和时间曲线。"
    )

if not selected_clusters:
    st.warning("至少选择一个运营簇。")
    st.stop()

cluster_view = clusters[clusters["cluster"].isin(selected_clusters)].copy()
annual_view = annual[
    (annual["year"] >= year_range[0]) & (annual["year"] <= year_range[1])
].copy()
metrics = metric_row(cluster_view, annual_view)

overview_tab, cluster_tab, time_tab, network_tab = st.tabs(
    ["总览", "分群", "时间", "网络"]
)

with overview_tab:
    left, middle, right, far = st.columns(4)
    left.metric("筛选商品数", f"{metrics['products']:,}")
    middle.metric("加权平均评分", f"{metrics['rating']:.3f}")
    right.metric("加权平均评论量", f"{metrics['reviews']:.1f}")
    far.metric("时间外推宏 F1", f"{metrics['time_f1']:.4f}" if metrics["time_f1"] else "—")

    st.caption(
        "时间 F1 为所选年份范围内 ≤2014→2015+ 的宏 F1 均值；"
        "商品数、评分、评论量、价格观测率按所选簇的商品数加权。"
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("模型对比")
        fig = px.bar(
            models,
            x="模型",
            y="宏 F1",
            color="模型",
            text=models["宏 F1"].map(lambda value: f"{value:.3f}"),
            color_discrete_sequence=["#5B8FF9", "#F6BD16", "#5AD8A6", "#E8684A"],
            title="模态消融：宏 F1（5 折 OOF）",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, yaxis_range=[0, 1], height=360)
        st.plotly_chart(fig, width="stretch")

    with chart_right:
        st.subheader("所选簇的商品规模")
        fig = px.bar(
            cluster_view,
            x="cluster_label",
            y="n_products",
            color="short_label",
            text="n_products",
            color_discrete_map=CLUSTER_COLORS,
            title="商品数（按运营簇）",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="", height=360)
        st.plotly_chart(fig, width="stretch")

    st.subheader("四簇总览")
    st.dataframe(
        cluster_view[
            [
                "cluster_label",
                "n_products",
                "mean_overall",
                "n_reviews",
                "verified_share",
                "price_observed",
                "mean_also_buy",
            ]
        ].rename(
            columns={
                "cluster_label": "簇",
                "n_products": "商品数",
                "mean_overall": "平均评分",
                "n_reviews": "平均评论量",
                "verified_share": "verified 比例",
                "price_observed": "价格观测率",
                "mean_also_buy": "also_buy 规模",
            }
        ),
        width="stretch",
        hide_index=True,
    )

with cluster_tab:
    if cluster_view.empty:
        st.info("当前筛选没有簇。")
    else:
        st.subheader("热度 × 评分 × 规模")
        fig = px.scatter(
            cluster_view,
            x="n_reviews",
            y="mean_overall",
            size="n_products",
            color="short_label",
            hover_name="cluster_label",
            text="short_label",
            log_x=True,
            color_discrete_map=CLUSTER_COLORS,
            labels={
                "n_reviews": "平均评论量（对数）",
                "mean_overall": "平均评分",
                "n_products": "商品数",
            },
            title="评论量与平均评分（气泡大小=商品数）",
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(showlegend=False, height=430)
        st.plotly_chart(fig, width="stretch")

        st.subheader("簇画像对比")
        metric = st.radio(
            "选择对比指标",
            ["平均评分", "平均评论量", "verified 比例", "价格观测率", "also_buy 规模"],
            horizontal=True,
        )
        column = {
            "平均评分": "mean_overall",
            "平均评论量": "n_reviews",
            "verified 比例": "verified_share",
            "价格观测率": "price_observed",
            "also_buy 规模": "mean_also_buy",
        }[metric]
        fig = px.bar(
            cluster_view,
            x="cluster_label",
            y=column,
            color="short_label",
            text=column,
            color_discrete_map=CLUSTER_COLORS,
            title=f"{metric}（按运营簇）",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="", height=380)
        st.plotly_chart(fig, width="stretch")

        st.caption(
            "业务读法：簇 1 高热度/共购密集，适合系列捆绑；簇 0 评分偏低、verified 高，适合质量排查；"
            "簇 2 是长尾，适合搜索/自动化；簇 3 是 Software 小段，价格观测率最高。"
        )

with time_tab:
    if annual_view.empty:
        st.info("当前年份范围没有数据。")
    else:
        st.subheader("逐年宏 F1")
        fig = px.line(
            annual_view,
            x="year",
            y="macro_f1",
            color="direction",
            markers=True,
            labels={"year": "评论年份", "macro_f1": "宏 F1", "direction": "方向"},
            title="时间外推：宏 F1 随年份变化",
        )
        fig.update_layout(height=380, xaxis=dict(dtick=1))
        st.plotly_chart(fig, width="stretch")

        st.subheader("逐年 PR-AUC")
        fig = px.line(
            annual_view,
            x="year",
            y="pr_auc",
            color="direction",
            markers=True,
            labels={"year": "评论年份", "pr_auc": "PR-AUC", "direction": "方向"},
            title="时间外推：PR-AUC 随年份变化",
        )
        fig.update_layout(height=360, xaxis=dict(dtick=1))
        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            annual_view[["direction", "year", "n", "positive_rate", "macro_f1", "pr_auc", "roc_auc"]].rename(
                columns={
                    "direction": "方向",
                    "year": "年份",
                    "n": "评论数",
                    "positive_rate": "正类率",
                    "macro_f1": "宏 F1",
                    "pr_auc": "PR-AUC",
                    "roc_auc": "ROC-AUC",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "正向 2015–2018 宏 F1 稳定在 0.90 左右；反向回测早期（2003–2006）明显更低，"
            "说明早期文本不能当同分布样本。"
        )

with network_tab:
    st.subheader("三张网络的结构差异")
    metric = st.radio(
        "网络指标",
        ["跨主品类边占比", "跨叶子边占比", "多边占比", "边权中位数"],
        horizontal=True,
    )
    fig = px.bar(
        networks,
        x="网络",
        y=metric,
        color="网络",
        text=metric,
        title=f"{metric}（also_buy / also_view / 共同评论人）",
        color_discrete_sequence=["#5B8FF9", "#F6BD16", "#5AD8A6"],
    )
    if metric in {"跨主品类边占比", "跨叶子边占比", "多边占比"}:
        fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        fig.update_layout(yaxis_tickformat=".0%")
    else:
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, width="stretch")

    st.subheader("边重叠")
    fig = px.bar(
        overlap,
        x="对比",
        y="Jaccard",
        text="Jaccard",
        title="Jaccard 重叠（越高越相似）",
        color_discrete_sequence=["#5B8FF9"],
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(height=360, yaxis_range=[0, 0.3])
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        networks.rename(
            columns={
                "跨主品类边占比": "跨主品类",
                "跨叶子边占比": "跨叶子",
                "多边占比": "多边",
                "边权中位数": "边权中位数",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "also_buy 与共同评论人网络的 Jaccard 仅 0.0102；推荐位网络比浏览/评论网络更窄，"
        "不能把 also_buy 当作真实购买网络。"
    )

st.divider()
st.caption(
    "数据来源：Amazon Review Data (2018)，UCSD；统计口径见 README 与 docs/。"
    "also_buy 是推荐位，不是真实订单；价格缺失 83.44%，价格只做事后画像。"
)
