#!/usr/bin/env python3
# app.py
import os, math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Output, Input, no_update, ctx, State
from fit_utils import fit_and_predict



# ==== 配置 ====
CSV_PATH   = "samples_mixed.csv"      # 参数点
OUTPUT_DIR = "output_maps_csv"            # 直方图输出目录
PREF_FMT   = "r{idx:04d}"             # 行号 -> 前缀（r0001, r0002, ...）

# ==== 配置 ====
# OUTPUT_DIR = "output_maps_csv"   # 原始 txt 的目录（回退用）

CACHE_DIR = "data_cache"         # data_prep.py --out 的目录
PARQUET_PATH = os.path.join(CACHE_DIR, "samples.parquet")
CSV_FALLBACK = os.path.join(CACHE_DIR, "samples.csv")
HIST_CACHE_DIR = os.path.join(CACHE_DIR, "hists")  # NPZ 缓存目录



# ==== 读数据 ====
# ==== 读数据（缓存优先）====
if os.path.exists(PARQUET_PATH):
    df = pd.read_parquet(PARQUET_PATH)
elif os.path.exists(CSV_FALLBACK):
    df = pd.read_csv(CSV_FALLBACK)
else:
    raise FileNotFoundError("No cached samples found (data_cache/samples.parquet or samples.csv). Run data_prep.py first.")

# 兜底：若预处理时没写 rid/row_index，就补上
if "row_index" not in df.columns or "rid" not in df.columns:
    df = df.reset_index(drop=True)
    df["row_index"] = df.index + 1
    df["rid"] = df["row_index"].apply(lambda i: f"r{i:04d}")


# ==== 3D 散点 ====
scatter = go.Scatter3d(
    x=df["kappa"], y=df["gamma"], z=df["s"],
    mode="markers",
    marker=dict(
        size=3,
        opacity=0.85,
        color=(df["kappa"]**2 + df["gamma"]**2)**-0.1,  # 颜色映射
        colorscale="Viridis",                   # 可以换成你喜欢的色盘
        colorbar=dict(title="k²+γ²")
    ),
    hovertemplate=(
        "idx=%{customdata[0]}<br>"
        "rid=%{customdata[1]}<br>"
        "k=%{x:.4f}<br>"
        "y=%{y:.4f}<br>"
        "s=%{z:.4f}<extra></extra>"
    ),
    customdata=np.c_[df["row_index"], df["rid"]],  # 提供 idx、rid 给回调使用
)

fig_scatter = go.Figure(scatter)
fig_scatter.update_layout(
    height=600, margin=dict(l=10, r=10, t=30, b=10),
    scene=dict(
        xaxis_title="kappa",
        yaxis_title="gamma",
        zaxis_title="s",
    ),
    title="Samples (hover/click a point → histogram on the right)"
)

# ==== Dash App ====
app = Dash(__name__)


app.layout = html.Div([
    # 顶层状态
    dcc.Store(id="locked-point", data=None),

    # 左：3D 散点图
    html.Div([
        dcc.Graph(id="scatter3d", figure=fig_scatter, clear_on_unhover=False),
    ], style={"width": "58%", "display": "inline-block", "verticalAlign": "top"}),

    # 右：唯一的一份控件 + 直方图
    html.Div([
        # 交互模式：hover / click
        dcc.RadioItems(
            id="mode",
            options=[{"label": "Hover 模式", "value": "hover"},
                     {"label": "Click 模式", "value": "click"}],
            value="hover", inline=True, style={"marginBottom": "6px"}
        ),

        # X 轴刻度：linear / log
        dcc.RadioItems(
            id="xscale",
            options=[{"label": "linear", "value": "linear"},
                     {"label": "log",    "value": "log"}],
            value="linear", inline=True, style={"marginBottom": "6px"}
        ),

        # X 轴范围模式：正常 / 固定 / 比例
        dcc.RadioItems(
            id="xlim-mode",
            options=[{"label": "正常", "value": "normal"},
                     {"label": "固定", "value": "fixed"},
                     {"label": "比例", "value": "percent"}],
            value="normal", inline=True, style={"margin": "6px 0"}
        ),

        # 是否叠加拟合
        dcc.Checklist(
            id="fit-on", options=[{"label":" 叠加拟合（LN+Pareto）", "value":"on"}],
            value=[], inline=True, style={"margin":"4px 0"}
        ),

        # 阈值百分位（用于拆分主体/尾部）
        dcc.Slider(
            id="fit-pct", min=85.0, max=99.5, step=0.5, value=95.0,
            marks={90:"90%", 95:"95%", 97.5:"97.5%", 99:"99%", 99.5:"99.5%"},
            tooltip={"placement":"bottom", "always_visible": False}
        ),


        # 固定范围滑条（仅 fixed 时显示）
        # html.Div(id="fixed-wrap", children=[
        #     dcc.RangeSlider(
        #         id="xlim-fixed",
        #         min=0.0, max=1.0, step="any",
        #         value=[0.0, 1.0],
        #         tooltip={"placement": "bottom", "always_visible": True}
        #     )
        # ], style={"display": "none", "marginBottom": "6px"}),
        html.Div(id="fixed-wrap", children=[
            dcc.RangeSlider(
                id="xlim-fixed",
                min=0.0, max=1.0, step=None,  # None 表示任意步长
                value=[0.0, 1.0],
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={"display": "none", "marginBottom": "6px"}),


        # 比例范围滑条（仅 percent 时显示）
        html.Div(id="percent-wrap", children=[
            dcc.RangeSlider(
                id="xlim-percent",
                min=0, max=100, step=1,
                value=[0, 100],
                marks={0: "0%", 25: "25%", 50: "50%", 75: "75%", 100: "100%"},
                tooltip={"placement": "bottom", "always_visible": False}
            )
        ], style={"display": "none", "marginBottom": "6px"}),

        html.Div(id="meta", style={"fontFamily":"monospace", "margin":"4px 0 8px 0"}),
        dcc.Graph(id="histplot", figure=go.Figure()),
    ], style={"width": "41%", "display": "inline-block", "verticalAlign": "top", "marginLeft": "1%"}),
])





from functools import lru_cache

@lru_cache(maxsize=4096)
def _load_hist_npz(rid: str):
    """从缓存 NPZ 读取 (mu, cnt, src, path)；没有则返回 None"""
    npz_path = os.path.join(HIST_CACHE_DIR, f"{rid}.npz")
    if not os.path.exists(npz_path):
        return None
    z = np.load(npz_path)
    mu  = z["mu"].astype(float)      # 已经是 μ（线性或 log->μ中点）
    cnt = z["cnt"].astype(float)
    src = "linear" if int(z["src"][0]) == 0 else "log"
    return mu, cnt, src, npz_path

@lru_cache(maxsize=4096)
def _load_hist_txt_linear(rid: str):
    path = os.path.join(OUTPUT_DIR, f"{rid}_ipm_mags_numpixels.txt")
    if not os.path.exists(path): return None
    arr = np.genfromtxt(path, usecols=(0,1))
    if arr.size == 0: return None
    arr = np.atleast_2d(arr)
    mu  = arr[:,0].astype(float) / 1000.0
    cnt = arr[:,1].astype(float)
    order = np.argsort(mu)
    return mu[order], cnt[order], "linear", path

@lru_cache(maxsize=4096)
def _load_hist_txt_log(rid: str):
    path = os.path.join(OUTPUT_DIR, f"{rid}_ipm_log_mags_numpixels.txt")
    if not os.path.exists(path): return None
    arr = np.genfromtxt(path, usecols=(0,1))
    if arr.size == 0: return None
    arr = np.atleast_2d(arr)
    b   = arr[:,0].astype(float)
    cnt = arr[:,1].astype(float)
    L   = b/100.0; dL=0.01
    mu  = np.exp(L + 0.5*dL)        # μ 中点
    order = np.argsort(mu)
    return mu[order], cnt[order], "log", path

def load_hist_for_rid(rid: str, xscale: str = "linear"):
    # 0) NPZ 缓存最快
    got = _load_hist_npz(rid)
    if got is None:
        # 1) 回退 txt（线性优先，其次 log）
        got = _load_hist_txt_linear(rid) or _load_hist_txt_log(rid)
    if got is None:
        fig = go.Figure()
        fig.update_layout(
            title=f"{rid}: histogram not found",
            xaxis_title="μ", yaxis_title="count",
            height=600, margin=dict(l=10, r=10, t=40, b=40)
        ); fig.update_xaxes(type=xscale)
        meta = f"{rid} • no histogram in cache or {OUTPUT_DIR}"
        return fig, meta

    mu, cnt, src, src_path = got
    if xscale == "log":
        m = mu > 0
        mu, cnt = mu[m], cnt[m]
    # 用 Scatter 更轻；若点数很多可以换 Scattergl
    fig = go.Figure(go.Scatter(x=mu, y=cnt, mode="lines"))
    fig.update_layout(
        title=f"{rid}: histogram ({'cached' if src_path.endswith('.npz') else 'raw'} {src})",
        xaxis_title="μ" if src=="linear" else "μ (from ln bins)",
        yaxis_title="count", height=600, margin=dict(l=10, r=10, t=40, b=40),
        uirevision="hist-sticky"
    ); fig.update_xaxes(type=xscale)
    meta = f"{rid} • source={os.path.basename(src_path)} • bins={len(mu)} • src={src}"
    return fig, meta





@app.callback(
    Output("fixed-wrap", "style"),
    Output("percent-wrap", "style"),
    Input("xlim-mode", "value"),
)
def _toggle_xlim_controls(mode):
    if mode == "fixed":
        return {"display": "block", "marginBottom": "6px"}, {"display": "none"}
    elif mode == "percent":
        return {"display": "none"}, {"display": "block", "marginBottom": "6px"}
    else:
        return {"display": "none"}, {"display": "none"}



    
def get_mu_for_rid(rid: str, xscale: str = "linear"):
    """返回当前 rid 的 μ 数组（已按 xscale==log 时过滤 μ>0 并排序）"""
    got = _load_hist_npz(rid) or _load_hist_txt_linear(rid) or _load_hist_txt_log(rid)
    if got is None:
        return None
    mu, cnt, src, src_path = got
    if xscale == "log":
        m = mu > 0
        mu = mu[m]
    mu = np.asarray(mu)
    mu.sort()
    return mu


def get_hover_idx_rid(hoverData):
    if not hoverData:
        return None, None
    pt = hoverData["points"][0]
    idx, rid = pt["customdata"]  # row_index, rid
    return int(idx), str(rid)


@app.callback(
    Output("locked-point", "data"),
    Input("scatter3d", "clickData"),
    State("locked-point", "data"),
    prevent_initial_call=True
)
def toggle_lock(clickData, locked_point):
    """点击相同点解锁，否则锁定新点"""
    if not clickData:
        return locked_point
    idx, rid = clickData["points"][0]["customdata"]
    new_point = (int(idx), str(rid))
    # 如果点击的是当前锁定点 → 取消锁定
    if locked_point == list(new_point):
        return None
    return list(new_point)

@app.callback(
    Output("histplot", "figure"),
    Output("meta", "children"),
    # 回填“固定”滑条的 min/max/value（每次换点时更新）
    Output("xlim-fixed", "min"),
    Output("xlim-fixed", "max"),
    Output("xlim-fixed", "value"),
    # 回填“比例”滑条的 value（每次换点时重置为 [0,100]）
    Output("xlim-percent", "value"),

    Input("scatter3d", "hoverData"),
    Input("scatter3d", "clickData"),
    Input("xscale", "value"),
    Input("mode", "value"),        # 你的 hover/click 行为模式
    Input("xlim-mode", "value"),   # 新增：xlim 模式
    Input("xlim-fixed", "value"),
    Input("xlim-percent", "value"),
    State("locked-point", "data"),
    Input("fit-on", "value"),
    Input("fit-pct", "value"),

)
def update_hist(hoverData, clickData, xscale, mode, xlim_mode, xlim_fixed, xlim_percent, locked_point):
    # 1) 选择数据来源（优先锁定；否则按照 hover/click 模式）
    if locked_point:
        idx, rid = locked_point
    elif mode == "click":
        if not clickData: return no_update, no_update, no_update, no_update, no_update, no_update
        idx, rid = clickData["points"][0]["customdata"]
    else:  # hover
        if not hoverData: return no_update, no_update, no_update, no_update, no_update, no_update
        idx, rid = hoverData["points"][0]["customdata"]

    # 2) 画图（用你已有的函数）
    fig, meta = load_hist_for_rid(rid, xscale=xscale)



        # === 拟合叠加（可选） ===
    if isinstance(fit_on, (list, tuple)) and ("on" in fit_on):
        # 取当前曲线的原始 μ 与 count —— 直接从缓存层拿（你已有 get_mu_for_rid 也行）
        # 复用 load_hist_for_rid 里同样的加载逻辑，或你已有 get_mu_for_rid_and_cnt()
        got = _load_hist_npz(rid) or _load_hist_txt_linear(rid) or _load_hist_txt_log(rid)
        if got is not None:
            mu_fit, cnt_fit, _, _ = got
            # 注：fit_and_predict 内部会做排序/过滤与单位换算
            fitres = fit_and_predict(mu_fit, cnt_fit, pct=float(fit_pct))
            if fitres is not None:
                x = fitres["curves"]["x"]
                y_bulk = fitres["curves"]["bulk"]
                y_tail = fitres["curves"]["tail"]
                y_mix  = fitres["curves"]["mix"]
                # 叠加三条曲线（你也可以只画 mix）
                fig.add_scatter(x=x, y=y_mix,  mode="lines", name="fit: mix",  line=dict(width=2))
                fig.add_scatter(x=x, y=y_bulk, mode="lines", name="fit: bulk", line=dict(width=1, dash="dot"))
                fig.add_scatter(x=x, y=y_tail, mode="lines", name="fit: tail", line=dict(width=1, dash="dash"))

                p = fitres["params"]
                meta += f" | fit μ0={p['mu0']:.3g}, lnμ~N({p['m_ln']:.3g},{p['s_ln']:.3g}), α={p['alpha']:.3g}, tail={p['mass_tail']*100:.1f}%"


    # 3) 计算当前直方图的 μ 范围，并准备回填滑条
    mu = get_mu_for_rid(rid, xscale=xscale)
    if mu is None or len(mu) == 0:
        # 找不到数据：只返回图和 meta，其他 no_update
        row = df.loc[df["rid"] == rid].iloc[0]
        meta2 = f"idx={int(idx)}  rid={rid}  k={row.kappa:.6f}  y={row.gamma:.6f}  s={row.s:.6f}"
        return fig, meta + " | " + meta2, no_update, no_update, no_update, no_update

    mu_min, mu_max = float(mu[0]), float(mu[-1])

    # 4) 根据 xlim 模式设置坐标范围
    if xlim_mode == "fixed" and xlim_fixed is not None:
        xmin, xmax = float(xlim_fixed[0]), float(xlim_fixed[1])
        if xscale == "log":
            # log 轴需要用 log10 的范围
            xmin = max(xmin, np.finfo(float).tiny)
            xmax = max(xmax, xmin * (1 + 1e-9))
            fig.update_xaxes(type="log", range=[np.log10(xmin), np.log10(xmax)])
        else:
            fig.update_xaxes(autorange=False, range=[xmin, xmax])

    elif xlim_mode == "percent" and xlim_percent is not None:
        pmin, pmax = float(xlim_percent[0]), float(xlim_percent[1])
        pmin = max(0.0, min(100.0, pmin))
        pmax = max(pmin, min(100.0, pmax))
        xmin = mu_min + (mu_max - mu_min) * (pmin / 100.0)
        xmax = mu_min + (mu_max - mu_min) * (pmax / 100.0)
        if xscale == "log":
            xmin = max(xmin, np.finfo(float).tiny)
            xmax = max(xmax, xmin * (1 + 1e-9))
            fig.update_xaxes(type="log", range=[np.log10(xmin), np.log10(xmax)])
        else:
            fig.update_xaxes(autorange=False, range=[xmin, xmax])
    else:
        # normal: 自适应
        fig.update_xaxes(autorange=True, type=xscale)

    # 5) 回填“固定”滑条的 min/max/value（让它跟当前直方图同步）
    fixed_min = mu_min
    fixed_max = mu_max
    # 若当前处于 fixed 模式就保持用户 value，否则重置为全范围
    fixed_value = xlim_fixed if (xlim_mode == "fixed" and xlim_fixed is not None) else [mu_min, mu_max]

    # 6) 回填“比例”滑条（非 percent 模式时重置为 [0,100]）
    percent_value = xlim_percent if (xlim_mode == "percent" and xlim_percent is not None) else [0, 100]

    # 7) meta 附加参数
    row = df.loc[df["rid"] == rid].iloc[0]
    meta2 = f"idx={int(idx)}  rid={rid}  k={row.kappa:.6f}  y={row.gamma:.6f}  s={row.s:.6f}"

    return fig, meta + " | " + meta2, fixed_min, fixed_max, fixed_value, percent_value




if __name__ == "__main__":
    app.run(debug=True, port=8050)


