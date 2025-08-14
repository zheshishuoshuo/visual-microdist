#!/usr/bin/env python3
# app.py
import os, math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Output, Input, no_update, ctx



# ==== 配置 ====
CSV_PATH   = "samples_mixed.csv"      # 参数点
OUTPUT_DIR = "output_maps_csv"            # 直方图输出目录
PREF_FMT   = "r{idx:04d}"             # 行号 -> 前缀（r0001, r0002, ...）

# ==== 读数据 ====
df = pd.read_csv(CSV_PATH).iloc[:970]
# 约定：行号从 1 开始映射到 r%04d（与你之前批跑脚本一致）
df = df.reset_index(drop=True)
df["row_index"] = df.index + 1
df["rid"] = df["row_index"].apply(lambda i: PREF_FMT.format(idx=i))

# ==== 3D 散点 ====
scatter = go.Scatter3d(
    x=df["kappa"], y=df["gamma"], z=df["s"],
    mode="markers",
    marker=dict(
        size=3,
        opacity=0.85,
        color=df["kappa"]**2 + df["gamma"]**2,  # 颜色映射
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
    html.Div([
        dcc.Graph(id="scatter3d", figure=fig_scatter, clear_on_unhover=False),
    ], style={"width": "58%", "display": "inline-block", "verticalAlign": "top"}),

    # html.Div([
    #     html.Div(id="meta", style={"fontFamily":"monospace", "margin":"4px 0 8px 0"}),
    #     dcc.Graph(id="histplot", figure=go.Figure(
    #         layout=go.Layout(
    #             title="Hover or click a point to load its histogram",
    #             xaxis_title="magnification μ (approx)",
    #             yaxis_title="count",
    #             height=600, margin=dict(l=10, r=10, t=40, b=40)
    #         )
    #     )),
    # ], style={"width": "41%", "display": "inline-block", "verticalAlign": "top", "marginLeft": "1%"}),





    # ... 你的 layout 中右侧容器里，hist 图上方加一个切换控件
    html.Div([
        dcc.RadioItems(
            id="xscale",
            options=[{"label": "linear", "value": "linear"},
                    {"label": "log",    "value": "log"}],
            value="linear",
            inline=True,
            style={"marginBottom": "6px"}
        ),
        html.Div(id="meta", style={"fontFamily":"monospace", "margin":"4px 0 8px 0"}),
        dcc.Graph(id="histplot", figure=go.Figure())
    ], style={"width": "41%", "display": "inline-block", "verticalAlign": "top", "marginLeft": "1%"})


])
def load_hist_for_rid(rid: str, xscale: str = "linear"):
    mags_path = os.path.join(OUTPUT_DIR, f"{rid}_ipm_mags_numpixels.txt")
    log_path  = os.path.join(OUTPUT_DIR, f"{rid}_ipm_log_mags_numpixels.txt")

    # 线性 μ 直方图
    if os.path.exists(mags_path):
        try:
            arr = np.loadtxt(mags_path)
            if arr.ndim == 1 and arr.size == 2:
                arr = arr.reshape(1, 2)
            mu  = arr[:, 0].astype(float) / 1000.0
            cnt = arr[:, 1].astype(float)
            if xscale == "log":
                m = mu > 0
                mu, cnt = mu[m], cnt[m]
            order = np.argsort(mu)
            mu, cnt = mu[order], cnt[order]
            fig = go.Figure(go.Scatter(x=mu, y=cnt, mode="lines"))
            fig.update_layout(
                title=f"{rid}: magnification histogram (linear bins)",
                xaxis_title="μ", yaxis_title="count",
                height=600, margin=dict(l=10, r=10, t=40, b=40)
            )
            fig.update_xaxes(type=xscale)
            meta = f"{rid} • source={os.path.basename(mags_path)} • bins={len(mu)}"
            return fig, meta
        except Exception:
            pass  # 回退到 log 版本

    # log 直方图 → 转 μ 中点再画
    if os.path.exists(log_path):
        try:
            arr = np.loadtxt(log_path)
            if arr.ndim == 1 and arr.size == 2:
                arr = arr.reshape(1, 2)
            b   = arr[:, 0].astype(float)
            cnt = arr[:, 1].astype(float)
            L   = b / 100.0
            dL  = 0.01
            mu_mid = np.exp(L + 0.5*dL)
            if xscale == "log":
                m = mu_mid > 0
                mu_mid, cnt = mu_mid[m], cnt[m]
            order = np.argsort(mu_mid)
            mu_mid, cnt = mu_mid[order], cnt[order]
            fig = go.Figure(go.Bar(x=mu_mid, y=cnt))
            fig.update_layout(
                title=f"{rid}: magnification histogram (log bins → μ mid)",
                xaxis_title="μ (from ln bins)", yaxis_title="count",
                height=600, margin=dict(l=10, r=10, t=40, b=40)
            )
            fig.update_xaxes(type=xscale)
            meta = f"{rid} • source={os.path.basename(log_path)} • bins={len(mu_mid)}"
            return fig, meta
        except Exception:
            pass

    # 占位
    fig = go.Figure()
    fig.update_layout(
        title=f"{rid}: histogram not found",
        xaxis_title="μ", yaxis_title="count",
        height=600, margin=dict(l=10, r=10, t=40, b=40)
    )
    fig.update_xaxes(type=xscale)
    meta = f"{rid} • no histogram file found under {OUTPUT_DIR}"
    return fig, meta


def get_hover_idx_rid(hoverData):
    if not hoverData:
        return None, None
    pt = hoverData["points"][0]
    idx, rid = pt["customdata"]  # row_index, rid
    return int(idx), str(rid)




@app.callback(
    Output("histplot", "figure"),
    Output("meta", "children"),
    Input("scatter3d", "clickData"),
    Input("scatter3d", "hoverData"),
    Input("xscale", "value"),
)
def update_hist(clickData, hoverData, xscale):
    triggered = ctx.triggered_id
    if triggered == "scatter3d" and hoverData:
        src = hoverData
    elif clickData:
        src = clickData
    else:
        return no_update, no_update

    idx, rid = src["points"][0]["customdata"]
    return load_hist_for_rid(rid, xscale)


if __name__ == "__main__":
    app.run(debug=True, port=8050)


