# fit_utils.py
import numpy as np
from functools import lru_cache
from scipy import special



# ========= 基础工具 =========

def weighted_quantile(x, w, q):
    """加权分位数 q∈[0,1]（x、w 一维且非负；返回按权重累计的分位点）"""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x, w = x[m], w[m]
    if x.size == 0 or w.sum() == 0:
        return np.nan
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w)
    t = q * cw[-1]
    k = np.searchsorted(cw, t, side="left")
    k = min(max(k, 0), x.size-1)
    return float(x[k])

def _norm_cdf(z):
    # 正态分布 CDF（erf 近似，足够拟合使用）
    return 0.5 * (1.0 + special.erf(z / np.sqrt(2.0)))

def lognormal_pdf(mu, m_ln, s_ln):
    mu = np.asarray(mu, float)
    mask = (mu > 0) & np.isfinite(mu)
    out = np.zeros_like(mu, dtype=float)
    z = (np.log(mu[mask]) - m_ln) / s_ln
    out[mask] = np.exp(-0.5*z*z) / (mu[mask] * s_ln * np.sqrt(2*np.pi))
    return out

def lognormal_cdf(mu, m_ln, s_ln):
    mu = np.asarray(mu, float)
    mask = (mu > 0) & np.isfinite(mu)
    out = np.zeros_like(mu, dtype=float)
    z = (np.log(mu[mask]) - m_ln) / s_ln
    out[mask] = _norm_cdf(z)
    return out

def pareto_pdf(mu, alpha, mu0):
    """标准 Pareto（下限 mu0），alpha>1 保证可积"""
    mu = np.asarray(mu, float)
    out = np.zeros_like(mu, dtype=float)
    m = (mu >= mu0) & np.isfinite(mu)
    if alpha <= 1:
        return out
    out[m] = (alpha - 1.0) * (mu0**(alpha - 1.0)) * (mu[m]**(-alpha))
    return out

# ========= 拟合：Lognormal（μ≤μ0） + Pareto（μ≥μ0） =========

def fit_lognorm_bulk(mu, cnt, mu0):
    """对 μ≤μ0 的主体做加权 MLE：返回 (m_ln, s_ln, mass_bulk, Z_bulk)
       mass_bulk = 主体权重（计数占比），Z_bulk = 截断归一化因子 Φ=P(μ≤μ0)
    """
    mu = np.asarray(mu, float); cnt = np.asarray(cnt, float)
    m = (mu > 0) & (mu <= mu0) & (cnt > 0)
    if m.sum() < 3:
        return None
    w = cnt[m]
    x = np.log(mu[m])
    m_ln = (w * x).sum() / w.sum()
    s2 = (w * (x - m_ln)**2).sum() / w.sum()
    s_ln = np.sqrt(max(s2, 1e-12))
    # 截断因子（Φ = P[μ<=μ0]）
    Z = float(lognormal_cdf(np.array([mu0]), m_ln, s_ln)[0])
    mass_bulk = float(cnt[m].sum() / cnt.sum())
    return m_ln, s_ln, mass_bulk, Z

def fit_pareto_tail(mu, cnt, mu0):
    """对 μ≥μ0 的尾部用加权 Hill 估计：alpha = 1 + W / sum(w*ln(mu/mu0))"""
    mu = np.asarray(mu, float); cnt = np.asarray(cnt, float)
    m = (mu >= mu0) & (cnt > 0)
    if m.sum() < 2:
        return None
    w = cnt[m]
    denom = (w * np.log(mu[m] / mu0)).sum()
    if denom <= 0:
        return None
    alpha = 1.0 + w.sum() / denom
    mass_tail = float(w.sum() / cnt.sum())
    return alpha, mass_tail

def build_piecewise_pdf(mu_grid, mu0, m_ln, s_ln, alpha, mass_bulk, mass_tail, Z_bulk):
    """返回分段混合 pdf： μ≤μ0 用截断LN， μ≥μ0 用Pareto；保证总质量=1"""
    pdf = np.zeros_like(mu_grid, dtype=float)
    left = mu_grid <= mu0
    right = ~left
    if left.any():
        f = lognormal_pdf(mu_grid[left], m_ln, s_ln)
        f = f / max(Z_bulk, 1e-12)
        pdf[left] = (1.0 - mass_tail) * f
    if right.any():
        pdf[right] = mass_tail * pareto_pdf(mu_grid[right], alpha, mu0)
    return pdf

# def predict_counts_curve(mu_grid, pdf, total_count):
#     """把 pdf 转为‘可与直方图对比’的计数曲线：count ≈ pdf * N * Δμ"""
#     if mu_grid.size < 2:
#         return np.zeros_like(mu_grid)
#     dmu = np.median(np.diff(mu_grid))
#     return pdf * float(total_count) * float(dmu)

def predict_counts_curve(mu_grid, pdf, total_count, bin_widths=None):
    if bin_widths is None:
        bin_widths = np.median(np.diff(mu_grid))
    return pdf * float(total_count) * bin_widths


# ========= 高层接口：给 Dash 用 =========

def fit_and_predict(mu, cnt, pct=95.0, grid_points=2000):
    """
    以百分位阈值 pct（如95）拆分主体/尾部，拟合并返回：
      dict(curves= { 'bulk':(x,y), 'tail':(x,y), 'mix':(x,y) },
           params= { 'mu0':..., 'm_ln':..., 's_ln':..., 'alpha':..., 'mass_tail':... })
    若任何一步失败，返回 None
    """
    mu = np.asarray(mu, float); cnt = np.asarray(cnt, float)
    ok = np.isfinite(mu) & np.isfinite(cnt) & (cnt >= 0)
    mu, cnt = mu[ok], cnt[ok]
    if mu.size < 8 or cnt.sum() <= 0:
        return None

    # 阈值：按计数的加权分位
    mu0 = weighted_quantile(mu, cnt, pct/100.0)
    if not np.isfinite(mu0):
        return None

    bulk = fit_lognorm_bulk(mu, cnt, mu0)
    tail = fit_pareto_tail(mu, cnt, mu0)
    if bulk is None or tail is None:
        return None

    m_ln, s_ln, mass_bulk, Z = bulk
    alpha, mass_tail = tail

    # 曲线
    # mu_grid = np.linspace(mu.min(), mu.max(), grid_points)
    # pdf_mix = build_piecewise_pdf(mu_grid, mu0, m_ln, s_ln, alpha, mass_tail, mass_tail, Z)
    # y_mix = predict_counts_curve(mu_grid, pdf_mix, cnt.sum())

    # # 单独画 bulk / tail（便于展示）
    # pdf_bulk = np.zeros_like(mu_grid); pdf_tail = np.zeros_like(mu_grid)
    # left = mu_grid <= mu0; right = ~left
    # if left.any():
    #     f = lognormal_pdf(mu_grid[left], m_ln, s_ln) / max(Z, 1e-12)
    #     pdf_bulk[left] = (1.0 - mass_tail) * f
    # if right.any():
    #     pdf_tail[right] = mass_tail * pareto_pdf(mu_grid[right], alpha, mu0)

    # y_bulk = predict_counts_curve(mu_grid, pdf_bulk, cnt.sum())
    # y_tail = predict_counts_curve(mu_grid, pdf_tail, cnt.sum())


    # 曲线
    mu_grid = np.linspace(mu.min(), mu.max(), grid_points)

    # ✅ mass_bulk = 1 - mass_tail，保证总质量守恒
    mass_bulk = 1.0 - mass_tail

    # 生成混合 PDF
    pdf_mix = build_piecewise_pdf(mu_grid, mu0, m_ln, s_ln, alpha, mass_bulk, mass_tail, Z)

    # 转成 count 曲线（和直方图一致）
    y_mix = predict_counts_curve(mu_grid, pdf_mix, cnt.sum())

    # 单独画 bulk / tail
    pdf_bulk = np.zeros_like(mu_grid)
    pdf_tail = np.zeros_like(mu_grid)

    left = mu_grid <= mu0
    right = ~left
    if left.any():
        f = lognormal_pdf(mu_grid[left], m_ln, s_ln) / max(Z, 1e-12)
        pdf_bulk[left] = mass_bulk * f
    if right.any():
        pdf_tail[right] = mass_tail * pareto_pdf(mu_grid[right], alpha, mu0)

    y_bulk = predict_counts_curve(mu_grid, pdf_bulk, cnt.sum())
    y_tail = predict_counts_curve(mu_grid, pdf_tail, cnt.sum())


    return dict(
        curves={
            "x": mu_grid,
            "bulk": y_bulk,
            "tail": y_tail,
            "mix": y_mix
        },
        params={
            "mu0": float(mu0),
            "m_ln": float(m_ln),
            "s_ln": float(s_ln),
            "alpha": float(alpha),
            "mass_tail": float(mass_tail)
        }
    )
