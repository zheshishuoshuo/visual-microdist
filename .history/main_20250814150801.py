#!/usr/bin/env python3
# app.py
import os, math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Output, Input, no_update

# ==== 配置 ====
CSV_PATH   = "samples_mixed.csv"      # 参数点
OUTPUT_DIR = "output_maps"            # 直方图输出目录
PREF_FMT   = "r{idx:04d}"             # 行号 -> 前缀（r0001, r0002, ...）

# ==== 读数据 ====
df = pd.read_csv(CSV_PATH)
# 约定：行号从 1 开始映射到 r%04d（与你之前批跑脚本一致）
df = df.reset_index(drop=True)
df["row_index"] = df.index + 1
df["rid"] = df["row_index"].apply(lambda i: PREF_FMT.format(idx=i))

# ==== 3D 散点 ====
scatter = go.Scatter3d(
    x=df["kappa"], y=df["gamma"], z=df["s"],
    mode="markers",
    marker=dict(size=3, opacity=0.85),
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

    html.Div([
        html.Div(id="meta", style={"fontFamily":"monospace", "margin":"4px 0 8px 0"}),
        dcc.Graph(id="histplot", figure=go.Figure(
            layout=go.Layout(
                title="Hover or click a point to load its histogram",
                xaxis_title="magnification μ (approx)",
                yaxis_title="count",
                height=600, margin=dict(l=10, r=10, t=40, b=40)
            )
        )),
    ], style={"width": "41%", "display": "inline-block", "verticalAlign": "top", "marginLeft": "1%"}),
])

def load_hist_for_rid(rid: str):
    """
    读取某个 rid 的直方图，优先使用线性 magnification 版本；
    若不存在则回退到 log 版本。返回 (fig, meta_text)
    """
    mags_path = os.path.join(OUTPUT_DIR, f"{rid}_ipm_mags_numpixels.txt")
    log_path  = os.path.join(OUTPUT_DIR, f"{rid}_ipm_log_mags_numpixels.txt")

    # 尝试线性 μ 直方图：两列（值已×1000, count）
    if os.path.exists(mags_path):
        try:
            arr = np.loadtxt(mags_path)
            if arr.ndim == 1 and arr.size == 2:
                arr = arr.reshape(1, 2)
            # 第一列 /1000 还原 μ
            mu = arr[:, 0].astype(float) / 1000.0
            cnt = arr[:, 1].astype(float)
            order = np.argsort(mu)
            mu, cnt = mu[order], cnt[order]
            fig = go.Figure(go.Bar(x=mu, y=cnt))
            fig.update_layout(
                title=f"{rid}: magnification histogram (linear bins)",
                xaxis_title="μ", yaxis_title="count",
                height=600, margin=dict(l=10, r=10, t=40, b=40)
            )
            # 对数轴可选：fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
            meta = f"{rid} • source={os.path.basename(mags_path)} • bins={len(mu)}"
            return fig, meta
        except Exception as e:
            pass  # 回退到 log 版本

    # 回退：log 版本（两列：bin_index, count），假设 ln μ，刻度 ×100
    if os.path.exists(log_path):
        try:
            arr = np.loadtxt(log_path)
            if arr.ndim == 1 and arr.size == 2:
                arr = arr.reshape(1, 2)
            b = arr[:, 0].astype(float)
            cnt = arr[:, 1].astype(float)
            L = b / 100.0                # bin 下边界
            dL = 0.01                    # 步长
            mu_mid = np.exp(L + 0.5*dL)  # 中点 μ
            order = np.argsort(mu_mid)
            mu_mid, cnt = mu_mid[order], cnt[order]
            fig = go.Figure(go.Bar(x=mu_mid, y=cnt))
            fig.update_layout(
                title=f"{rid}: magnification histogram (log bins → μ mid)",
                xaxis_title="μ (from ln bins)", yaxis_title="count",
                height=600, margin=dict(l=10, r=10, t=40, b=40)
            )
            meta = f"{rid} • source={os.path.basename(log_path)} • bins={len(mu_mid)}"
            return fig, meta
        except Exception as e:
            pass

    # 都没有：返回占位
    fig = go.Figure()
    fig.update_layout(
        title=f"{rid}: histogram not found",
        xaxis_title="μ", yaxis_title="count",
        height=600, margin=dict(l=10, r=10, t=40, b=40)
    )
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
    Input("scatter3d", "hoverData"),
    Input("scatter3d", "clickData"),
)
def update_hist(hoverData, clickData):
    # 优先用点击；没有点击则用 hover
    src = clickData or hoverData
    idx, rid = get_hover_idx_rid(src)
    if rid is None:
        return no_update, ""
    fig, meta = load_hist_for_rid(rid)
    # 附带参数提示
    row = df.loc[df["rid"] == rid].iloc[0]
    meta2 = f"idx={idx}  rid={rid}  k={row.kappa:.6f}  y={row.gamma:.6f}  s={row.s:.6f}"
    return fig, meta + " | " + meta2

if __name__ == "__main__":
    app.run(debug=True, port=8050)


