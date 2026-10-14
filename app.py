# -*- coding: utf-8 -*-
"""潮涌有数 · 温州城市内涝风险筛查与成因归因（Streamlit 应用）

运行：
    streamlit run app.py
或使用同目录的 启动Demo.bat

数据说明 / 安全边界见页面内「方法与数据」「局限与安全边界」标签页。
"""
import io
import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_JSON = os.path.join(HERE, "demo_data.json")
TOP_JSON = os.path.join(HERE, "demo_top20.json")

st.set_page_config(page_title="潮涌有数 · 温州内涝风险筛查", page_icon="🌊", layout="wide")

# ----------------------------------------------------------------- 样式
st.markdown("""
<style>
.block-container{padding-top:2.2rem;padding-bottom:3rem;max-width:1500px}
h1,h2,h3{color:#1F4E79}
.kpi{background:#fff;border-radius:10px;padding:14px 16px;text-align:center;
     box-shadow:0 2px 10px rgba(31,78,121,.10);border-top:4px solid #2E75B6}
.kpi .v{font-size:27px;font-weight:800;color:#1F4E79;line-height:1.15}
.kpi .l{font-size:12px;color:#66788A;margin-top:5px}
.warn{background:#FFF6F6;border-left:4px solid #C00000;padding:11px 15px;
      border-radius:0 8px 8px 0;color:#5A1A1A;font-size:13px;margin:10px 0}
.info{background:#F2F8FE;border-left:4px solid #2E75B6;padding:11px 15px;
      border-radius:0 8px 8px 0;color:#1B4F72;font-size:13px;margin:10px 0}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_all():
    with open(DATA_JSON, encoding="utf-8") as f:
        d = json.load(f)
    with open(TOP_JSON, encoding="utf-8") as f:
        t = json.load(f)
    m = d["meta"]
    df = pd.DataFrame(d["candidates"], columns=[
        "lon", "lat", "score", "elev", "dep", "water", "under", "road"])
    nx, ny, cell = m["nx"], m["ny"], m["cell"]
    lon0, lat0 = m["bbox"][0], m["bbox"][1]
    lons = lon0 + (np.arange(nx) + 0.5) * cell
    lats = lat0 + (np.arange(ny) + 0.5) * cell
    raster_score = np.array(d["raster_score"]).reshape(ny, nx)
    raster_elev = np.array(d["raster_elev"]).reshape(ny, nx)
    return m, df, pd.DataFrame(t), lons, lats, raster_score, raster_elev


M, DF, TOP, LONS, LATS, RSCORE, RELEV = load_all()

# ----------------------------------------------------------------- 头部
st.markdown("# 🌊 潮涌有数 · 温州城市内涝风险筛查与成因归因")
st.markdown(
    "**AI4Climate 全球青年气候解决方案挑战赛** ｜ 推荐方向 05 · 中国东南地区 · 台风与城市内涝韧性  \n"
    "参赛团队：施韵 3260104236　｜　郑源锦 3260104217　｜　金运梓 3260101622"
)
c1, c2, c3, c4 = st.columns(4)
for col, v, l in zip(
        [c1, c2, c3, c4],
        [f"{M['nx']}×{M['ny']}", "14,105", "2,438", f"{M['n_candidates']:,}"],
        ["风险栅格分辨率", "OSM 道路（条）", "OSM 水系（条）", "候选点位（路网栅格）"]):
    col.markdown(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>',
                 unsafe_allow_html=True)

st.markdown("")
st.markdown(
    '<div class="warn"><b>请注意：</b>本应用展示的是<b>基于公开数据的筛查结果</b>，'
    '风险权重按文献与官方成因表述设定，<b>尚未使用官方易涝点坐标进行校验</b>，'
    '不构成任何形式的灾害预警。</div>', unsafe_allow_html=True)

tab_map, tab_rank, tab_model, tab_method, tab_limit = st.tabs(
    ["① 风险地图与筛选", "② 风险排序与成因归因", "③ 权重实验（模型可调）",
     "④ 方法与数据", "⑤ 局限与安全边界"])

# ================================================================= ① 地图
with tab_map:
    left, right = st.columns([1, 3])
    with left:
        st.markdown("#### 筛选条件")
        f_score = st.slider("风险指数下限", 0.0, 1.0, 0.0, 0.01)
        f_elev = st.slider("高程上限（米）", 0, 60, 60, 1)
        f_water = st.slider("距水体上限（km）", 0.0, 5.0, 5.0, 0.1)
        f_dep = st.slider("相对低洼度下限（米）", 0.0, 20.0, 0.0, 0.5)
        f_under = st.checkbox("仅显示下穿 / 隧道结构", value=False)
        st.markdown("#### 底图")
        base = st.radio("图层", ["筛查风险指数", "高程"], index=0, label_visibility="collapsed")
        show_pts = st.checkbox("显示候选点位", value=True)
        show_top = st.checkbox("标记 Top-20", value=True)
        topn = st.number_input("地图显示前 N 个点位", 10, 3000, 300, 10)

    d = DF.copy()
    d = d[(d["score"] >= f_score) & (d["elev"] <= f_elev) &
          (d["water"] <= f_water) & (d["dep"] >= f_dep)]
    if f_under:
        d = d[d["under"] == 1]
    d = d.sort_values("score", ascending=False).reset_index(drop=True)

    with right:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("命中点位", f"{len(d):,}")
        k2.metric("平均风险", f"{d['score'].mean():.3f}" if len(d) else "—")
        k3.metric("其中下穿", f"{int(d['under'].sum()) if len(d) else 0} 个")
        k4.metric("平均高程", f"{d['elev'].mean():.1f} m" if len(d) else "—")

        # ---- 地图（Altair，无需地图瓦片，纯离线可用）
        rast = RSCORE if base == "筛查风险指数" else RELEV
        grid = pd.DataFrame({
            "lon": np.repeat(LONS, len(LATS)),
            "lat": np.tile(LATS, len(LONS)),
            "v": rast.T.ravel(),
        })
        if base == "筛查风险指数":
            grid = grid[grid["v"] > 0.06]
            cmap = alt.Scale(scheme="orangered", domain=[0, 1])
            ttl = "v"
        else:
            cmap = alt.Scale(scheme="tealblues", domain=[0, 150])
            ttl = "v"
        layers = [alt.Chart(grid).mark_square(size=13).encode(
            x=alt.X("lon:Q", title="经度", scale=alt.Scale(domain=[LONS[0] - .01, LONS[-1] + .01])),
            y=alt.Y("lat:Q", title="纬度", scale=alt.Scale(domain=[LATS[0] - .01, LATS[-1] + .01])),
            color=alt.Color("v:Q", scale=cmap, title=("风险指数" if base == "筛查风险指数" else "高程(m)")),
            tooltip=[alt.Tooltip("lon:Q", format=".4f"), alt.Tooltip("lat:Q", format=".4f"),
                     alt.Tooltip("v:Q", format=".3f")],
        ).properties(height=560)]

        if show_pts and len(d):
            pts = d.head(int(topn))
            layers.append(alt.Chart(pts).mark_circle(size=9, opacity=.85).encode(
                x="lon:Q", y="lat:Q",
                color=alt.Color("score:Q",
                                scale=alt.Scale(scheme="orangered", domain=[0, 1]),
                                legend=None),
                tooltip=[alt.Tooltip("lon:Q", format=".4f"), alt.Tooltip("lat:Q", format=".4f"),
                         alt.Tooltip("score:Q", title="风险", format=".3f"),
                         alt.Tooltip("elev:Q", title="高程m"), alt.Tooltip("dep:Q", title="低洼m"),
                         alt.Tooltip("water:Q", title="距水km"),
                         alt.Tooltip("under:Q", title="下穿")]))

        if show_top:
            tp = TOP.copy()
            layers.append(alt.Chart(tp).mark_point(
                size=210, filled=False, stroke="#C00000", strokeWidth=2.4).encode(
                x="lon:Q", y="lat:Q",
                tooltip=[alt.Tooltip("rank:Q", title="Top"), alt.Tooltip("score:Q", format=".3f"),
                         alt.Tooltip("lon:Q", format=".4f"), alt.Tooltip("lat:Q", format=".4f")]))

        if f_under or (show_pts and "under" in d.columns):
            und = DF[DF["under"] == 1]
            layers.append(alt.Chart(und).mark_point(
                shape="diamond", size=42, color="#4FC3F7", opacity=.9).encode(
                x="lon:Q", y="lat:Q",
                tooltip=[alt.Tooltip("lon:Q", format=".4f"), alt.Tooltip("lat:Q", format=".4f")]))

        st.altair_chart(alt.layer(*layers).interactive(), use_container_width=True)
        st.caption("滚轮/框选可缩放；悬停查看数值。底图为栅格，非在线地图瓦片，因此完全离线可用。")

        st.download_button(
            "⬇ 导出当前筛选结果 CSV",
            data=("\ufeff" + d.to_csv(index=False, encoding="utf-8")).encode("utf-8"),
            file_name="chaoyong_screening_filtered.csv", mime="text/csv")

# ================================================================= ② 排序与归因
with tab_rank:
    st.markdown("#### 风险排序（可排序、可搜索）")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        q = st.text_input("按坐标搜索（输入经度或纬度片段）", "")
    with c2:
        min_s = st.slider("只看风险 ≥", 0.0, 1.0, 0.0, 0.01)
    with c3:
        only_und = st.checkbox("只看下穿结构")

    t = DF[DF["score"] >= min_s]
    if only_und:
        t = t[t["under"] == 1]
    if q:
        t = t[t["lon"].astype(str).str.contains(q) | t["lat"].astype(str).str.contains(q)]
    t = t.sort_values("score", ascending=False).reset_index(drop=True)
    t.insert(0, "排名", np.arange(1, len(t) + 1))
    t["下穿"] = np.where(t["under"] == 1, "是", "否")
    show = t.head(500)[["排名", "lon", "lat", "score", "elev", "dep", "water", "road", "下穿"]]
    show.columns = ["排名", "经度", "纬度", "风险指数", "高程(m)", "低洼(m)",
                    "距水(km)", "道路密度(km)", "下穿"]
    st.dataframe(show, use_container_width=True, height=360, hide_index=True)
    st.caption(f"共 {len(t):,} 条命中；表格最多显示前 500 条，可用上方搜索缩小范围。")

    st.markdown("#### 单点成因归因")
    if len(t):
        sel = st.selectbox(
            "选择点位（按排名，默认第 1 名）",
            options=list(range(min(len(t), 200))),
            format_func=lambda i: (f"#{t.loc[i,'排名']}　{t.loc[i,'lon']:.5f}°E, "
                                   f"{t.loc[i,'lat']:.5f}°N　风险 {t.loc[i,'score']:.4f}"))
        row = t.loc[sel]
    else:
        st.info("当前条件下没有点位。")
        row = None

    if row is not None:
        W = dict(dep=0.34, water=0.24, conc=0.18, road=0.10)
        gate = 1.0 if row["elev"] < 5 else max(0.15, 1 - (row["elev"] - 5) / 20)
        boost = 2.2 if row["under"] == 1 else 1.0
        depn = min(max(row["dep"] / 3, 0), 1)
        wnear = float(np.exp(-min(row["water"], 5) / 1.5))
        roadn = min(max(row["road"] / 0.6, 0), 1)
        conc = min(max(gate * 0.5, 0), 1)
        contrib = pd.DataFrame({
            "因子": ["低洼（相对周边）", "近水体", "地形汇水（凹形）", "道路暴露度"],
            "贡献": [W["dep"] * depn * gate * boost, W["water"] * wnear * gate * boost,
                     W["conc"] * conc * gate * boost, W["road"] * roadn * gate * boost],
            "颜色": ["#C00000", "#2E75B6", "#ED7D31", "#548235"],
        }).sort_values("贡献")

        a, b = st.columns([1, 1])
        with a:
            st.markdown("##### 点位属性")
            st.dataframe(pd.DataFrame({
                "属性": ["筛查风险指数", "高程", "相对低洼度", "到水体距离",
                         "下穿/隧道结构", "道路密度", "低地门控系数"],
                "数值": [f"{row['score']:.4f}", f"{row['elev']:.1f} m", f"{row['dep']:.2f} m",
                         f"{row['water']:.2f} km", "是（×2.2 增益）" if row["under"] == 1 else "否",
                         f"{row['road']:.2f} km", f"{gate:.2f}"],
            }), use_container_width=True, hide_index=True, height=270)
        with b:
            st.markdown("##### 因子贡献")
            st.altair_chart(
                alt.Chart(contrib).mark_bar(size=26).encode(
                    x=alt.X("贡献:Q", title="贡献值"),
                    y=alt.Y("因子:N", sort="-x", title=""),
                    color=alt.Color("颜色:N", scale=None, legend=None),
                    tooltip=["因子", alt.Tooltip("贡献:Q", format=".4f")],
                ).properties(height=270), use_container_width=True)
        st.markdown(
            '<div class="warn">该点位为<b>筛查结果</b>，尚未经官方易涝点坐标校验。'
            '因子贡献按固定权重计算，<b>不代表模型学到的真实因果贡献</b>。</div>',
            unsafe_allow_html=True)

    st.markdown("#### 特征分布（真实数据）")
    cc = st.columns(3)
    for col, (k, lab, dom) in zip(cc, [
            ("score", "筛查风险指数", [0, 1]), ("elev", "高程（米）", [0, 40]),
            ("water", "到水体距离（km）", [0, 5])]):
        col.altair_chart(
            alt.Chart(DF).mark_bar().encode(
                x=alt.X(f"{k}:Q", bin=alt.Bin(maxbins=30), title=lab, scale=alt.Scale(domain=dom)),
                y=alt.Y("count():Q", title="点位数量"),
            ).properties(height=220, title=lab), use_container_width=True)
    st.markdown(
        '<div class="info"><b>关键发现：</b>92.1% 的候选点距水体不足 1 公里——温州是典型水网城市，'
        '「靠近水体」几乎无区分度。我们据此把主导因子从「近水」改为「地形低洼度」。</div>',
        unsafe_allow_html=True)

# ================================================================= ③ 权重实验
with tab_model:
    st.markdown("#### 权重实验：自己调权重，看排序怎么变")
    st.markdown(
        "这是本方案「可解释」的体现——你可以直接改动四个因子的权重，"
        "应用会**实时重算全城 8,509 个点位的风险排序**，并对比与默认排序的差异。"
    )
    st.markdown(
        '<div class="info">默认权重来自温州内涝三大成因的官方论述：'
        '地形汇水（低山丘陵占 62.14%、平原仅 21%）、感潮顶托、管网能力。'
        '权重是<b>专家先验</b>，不是从标签学出来的。</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    w_dep = c1.slider("低洼度权重", 0.0, 1.0, 0.34, 0.01)
    w_wat = c2.slider("近水体权重", 0.0, 1.0, 0.24, 0.01)
    w_con = c3.slider("地形汇水权重", 0.0, 1.0, 0.18, 0.01)
    w_rod = c4.slider("道路暴露度权重", 0.0, 1.0, 0.10, 0.01)

    total = w_dep + w_wat + w_con + w_rod
    if total <= 0:
        st.error("请至少给一个因子大于 0 的权重。")
    else:
        e = DF["elev"].values
        gate = np.where(e < 5, 1.0, np.clip(1 - (e - 5) / 20, 0.15, 1.0))
        gate = np.where(e > 40, 0.0, gate)
        boost = np.where(DF["under"].values == 1, 2.2, 1.0)
        depn = np.clip(DF["dep"].values / 3, 0, 1)
        wnear = np.exp(-np.clip(DF["water"].values, 0, 5) / 1.5)
        roadn = np.clip(DF["road"].values / 0.6, 0, 1)
        conc = np.clip(gate * 0.5, 0, 1)
        raw = (w_dep * depn + w_wat * wnear + w_con * conc + w_rod * roadn) * gate * boost
        raw = (raw - raw.min()) / (np.ptp(raw) + 1e-9)
        cmp = DF[["lon", "lat", "score"]].copy()
        cmp["新风险"] = raw
        cmp["新排名"] = cmp["新风险"].rank(ascending=False, method="first").astype(int)
        cmp["原排名"] = cmp["score"].rank(ascending=False, method="first").astype(int)
        cmp["排名变化"] = cmp["原排名"] - cmp["新排名"]

        k1, k2, k3 = st.columns(3)
        sp = cmp[["score", "新风险"]].corr(method="spearman").iloc[0, 1]
        k1.metric("与原排序的 Spearman 相关", f"{sp:.3f}")
        k2.metric("新 Top-20 仍在原 Top-20 内", f"{int((cmp.nsmallest(20,'新排名')['原排名'] <= 20).sum())} / 20")
        k3.metric("上升最多的点位", f"+{int(cmp['排名变化'].max())} 位")

        a, b = st.columns(2)
        with a:
            st.markdown("##### 新权重下的 Top-20")
            top20 = cmp.nsmallest(20, "新排名")[["新排名", "lon", "lat", "新风险", "排名变化"]]
            top20.columns = ["排名", "经度", "纬度", "新风险", "较原排名"]
            st.dataframe(top20, use_container_width=True, hide_index=True, height=420)
        with b:
            st.markdown("##### 排名变化最大的点位")
            chg = cmp.reindex(cmp["排名变化"].abs().sort_values(ascending=False).index).head(20)
            chg = chg[["lon", "lat", "原排名", "新排名", "排名变化"]]
            chg.columns = ["经度", "纬度", "原排名", "新排名", "变化"]
            st.dataframe(chg, use_container_width=True, hide_index=True, height=420)

        st.altair_chart(
            alt.Chart(cmp).mark_circle(size=8, opacity=.5, color="#1F4E79").encode(
                x=alt.X("score:Q", title="原风险指数", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("新风险:Q", title="新权重风险指数", scale=alt.Scale(domain=[0, 1])),
                tooltip=["lon", "lat", alt.Tooltip("score:Q", format=".3f"),
                         alt.Tooltip("新风险:Q", format=".3f")],
            ).properties(height=330, title="原排序 vs 新权重排序（点越偏离对角线，说明该点位对权重越敏感）"),
            use_container_width=True)

# ================================================================= ④ 方法
with tab_method:
    st.markdown("### 这个应用在展示什么")
    st.markdown(
        "它把我们已经跑通的**真实数据筛查流程**做成可交互应用：加载温州城区路网栅格候选点位"
        "及其真实空间特征，按可调权重的风险指数排序，并输出成因归因。\n\n"
        "**它不是一个训练好的预测模型。** 官方 27 处下穿隧道的逐条坐标未公开，我们拿不到标签，"
        "因此无法训练有监督模型、也无法报告真实精度。"
    )
    st.markdown("### 数据来源")
    st.dataframe(pd.DataFrame({
        "数据": ["路网 / 水系 / 铁路 / 下穿候选", "高程", "研究范围", "官方口径"],
        "来源": ["OpenStreetMap（Overpass API，ODbL）",
                 "Open-Meteo Elevation API（基于 Copernicus DEM GLO-90）",
                 "120.52–120.90°E, 27.87–28.10°N（鹿城/龙湾/瓯海建成区）",
                 "温州市政府网 2026-07-12；温州市公用集团排水公司 2026-08-11"],
        "规模": ["17,098 way / 147,744 几何节点", "3,619 个采样点",
                 "约 30 km × 25 km", "29 处低洼点 / 87 处实际积水点"],
    }), use_container_width=True, hide_index=True)
    st.markdown("### 风险指数定义")
    st.code(
        "风险 = 低地门控(高程) × [ 0.34×低洼度 + 0.24×近水体 + 0.18×地形汇水 + 0.10×道路暴露度 ]\n"
        "       × (1 + 1.2 × 下穿标记)\n\n"
        "低地门控：高程 < 5 m → 1.0；25 m → 0.15；> 40 m → 0",
        language="text")
    st.markdown(
        "**为什么用门控而不是简单加权**：温州 62% 是山地（官方数字）。"
        "山间谷底虽然「相对低洼」，但不会以城市平原的方式积水。"
        "我们第一版模型曾把瓯海山区谷底（低洼 16.7 m 但高程仅 4 m）判为高风险，加入门控后才修正。"
    )
    st.markdown("### 两个被数据推翻的假设")
    st.markdown(
        "1. **近水特征几乎无区分度**：92.1% 的候选点距水体不足 1 公里 → 改以地形低洼度为主导因子。\n"
        "2. **OSM 无法可靠识别下穿隧道**：Top-20 初期仅 20% 被标为下穿，而官方口径是 27/29（93%）；"
        "实测 `tunnel=culvert` 在城区返回 0 条 → 必须以政府清单为准。"
    )

# ================================================================= ⑤ 局限
with tab_limit:
    st.markdown("### 已知局限（必须与结果同时呈现）")
    for s in [
        "**未经官方坐标校验**：风险权重来自文献与官方成因表述，不是从标签学出来的。"
        "官方 27 处下穿隧道名单未逐条公开坐标，这是当前唯一的质量缺口。",
        "**DEM 分辨率 90 米**：无法捕捉小于 90 米尺度的微地形（如单个下穿隧道的坡度）。",
        "**无管网内部数据**：管径、埋深、流向不可得，排水能力只能用道路等级与低洼度间接表达。",
        "**无时间维度**：本阶段为静态筛查，未接入降雨时序与台风路径。",
        "**低洼度受 DEM 平滑影响**：2 km 邻域窗口对窄而深的下穿结构偏保守。",
        "**社交媒体信息存在伪造风险**：2026 年 7 月已有官方辟谣案例，多模态标签须两源以上交叉核验。",
    ]:
        st.markdown(f"- {s}")

    st.markdown("### 安全边界（真实使用中的红线）")
    for s in [
        "定位为**辅助决策**，不是自动预警；只输出相对风险排序与置信度，不输出确定性水深结论。",
        "**官方 29 处规程点位永久置顶**，模型只做增量补充，绝不削减既有防线。",
        "低置信度点位标注「建议人工复核」，不自动进入设备前置建议。",
        "公众版仅作科普性风险提示，并强制标注「请以官方预警为准」。",
        "禁止用于责任认定或工程验收等需要精确数值的场景。",
    ]:
        st.markdown(f"- {s}")

    st.markdown("### 预实验已完成的结论（合成空间场，真值已知）")
    st.dataframe(pd.DataFrame({
        "结论": ["随机划分会高估性能", "K=29 上官方清单更强", "K=87 上模型更强",
                 "标签质量 > 标签数量"],
        "数据": ["随机 K 折 AUC 0.897 vs 空间分块 0.878（虚高约 0.019）",
                 "清单 0.38 vs 模型 0.17", "模型 0.51 vs 清单 0.36（相对 +42%）",
                 "标签从 20 增至 200，Recall 反从 0.46 降到 0.37"],
        "对方案的影响": ["全面采用空间分块交叉验证", "官方 29 点永久置顶兜底",
                         "模型定位为增量补充", "扩样只采纳「有实际积水证据」的点位"],
    }), use_container_width=True, hide_index=True)
    st.markdown(
        '<div class="warn">以上为<b>合成空间场</b>上的方法验证结果，用于检验评估协议与设计决策，'
        '<b>不代表温州实测精度</b>。</div>', unsafe_allow_html=True)
