#!/usr/bin/env python3
# app.py
import os, math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Output, Input, no_update, ctx, State



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
        color=(df["kappa"]**2 + df["gamma"]**2)**-0.,  # 颜色映射
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
    dcc.Store(id="locked-point", data=None),  # 顶层放一个隐藏状态

    html.Div([
        dcc.Graph(id="scatter3d", figure=fig_scatter, clear_on_unhover=False),
    ], style={"width": "58%", "display": "inline-block", "verticalAlign": "top"}),

    html.Div([
        dcc.RadioItems(
            id="xscale",
            options=[{"label": "linear", "value": "linear"},
                     {"label": "log",    "value": "log"}],
            value="linear", inline=True, style={"marginBottom": "6px"}
        ),
        html.Div(id="meta", style={"fontFamily":"monospace", "margin":"4px 0 8px 0"}),
        dcc.Graph(id="histplot", figure=go.Figure()),
    ], style={"width": "41%", "display": "inline-block", "verticalAlign": "top", "marginLeft": "1%"}),

    dcc.RadioItems(
        id="mode",
        options=[
            {"label": "Hover 模式", "value": "hover"},
            {"label": "Click 模式", "value": "click"},
        ],
        value="hover",
        inline=True
    ),

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
    Input("scatter3d", "hoverData"),
    Input("scatter3d", "clickData"),
    Input("xscale", "value"),
    Input("mode", "value"),   # 新增
)
def update_hist(hoverData, clickData, xscale, mode):
    if mode == "click":
        if not clickData:
            return no_update, no_update
        idx, rid = clickData["points"][0]["customdata"]
    else:  # hover 模式
        if not hoverData:
            return no_update, no_update
        idx, rid = hoverData["points"][0]["customdata"]

    fig, meta = load_hist_for_rid(rid, xscale)
    row = df.loc[df["rid"] == rid].iloc[0]
    meta2 = f"idx={int(idx)}  rid={rid}  k={row.kappa:.6f}  y={row.gamma:.6f}  s={row.s:.6f}"
    return fig, meta + " | " + meta2



if __name__ == "__main__":
    app.run(debug=True, port=8050)


