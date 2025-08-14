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











from scipy.optimize import minimize

def fit_additive_lognorm_pareto_counts(mu, cnt, bin_widths, pct_for_mu_min=95.0):
    """
    两段叠加：counts ≈ N * Δμ * [(1-w) LN + w Pareto]
    输入：
      mu:           直方图中心 (array)
      cnt:          直方图计数 (array)
      bin_widths:   每个 bin 对应的 Δμ_i (array，线性=常数 1/1000；log 见 main)
      pct_for_mu_min: 以加权分位数确定 Pareto 的 μ_min（默认 95%）
    返回：
      dict(curves={'x':mu, 'mix':y, 'ln':y_ln, 'pareto':y_pa},
           params={'w':..., 'm_ln':..., 's_ln':..., 'alpha':..., 'mu_min':...})
      或 None（拟合失败）
    """
    mu   = np.asarray(mu, float)
    cnt  = np.asarray(cnt, float)
    bw   = np.asarray(bin_widths, float)
    ok = np.isfinite(mu) & np.isfinite(cnt) & np.isfinite(bw) & (bw > 0)
    mu, cnt, bw = mu[ok], cnt[ok], bw[ok]
    if mu.size < 8 or cnt.sum() <= 0:
        return None

    N = float(cnt.sum())
    # μ_min 用加权分位数初始化（也可让用户通过滑条控制）
    mu_min0 = max(1e-6, weighted_quantile(mu, cnt, pct_for_mu_min/100.0))
    # 初值：先用未截断 LN 的矩来初始化 m, s；w 初始化为尾部质量占比；alpha 给个温和初值
    x = np.log(np.clip(mu, 1e-12, None))
    wts = cnt
    m0 = (wts * x).sum() / wts.sum()
    s0 = np.sqrt(max((wts * (x - m0)**2).sum() / wts.sum(), 1e-3))
    w0 = min(0.4, float((cnt[mu >= mu_min0].sum() / N)))   # 初始尾部权重
    a0 = 3.0

    # 变量向量：theta = [logit(w), m_ln, log(s_ln), log(alpha-1), log(mu_min - mu_lo)]
    mu_lo = float(np.min(mu))
    def pack(w, m_ln, s_ln, alpha, mu_min):
        z0 = np.log(w/(1-w + 1e-12) + 1e-12)
        return np.array([z0, m_ln, np.log(s_ln), np.log(alpha-1.0), np.log(mu_min - mu_lo + 1e-9)], float)
    def unpack(th):
        z0, m_ln, ls, la1, lmm = th
        w = 1.0/(1.0 + np.exp(-z0))       # (0,1)
        s_ln = np.exp(ls)                 # >0
        alpha = 1.0 + np.exp(la1)         # >1
        mu_min = mu_lo + np.exp(lmm)      # >= mu_lo
        return w, m_ln, s_ln, alpha, mu_min

    def nll(th):
        w, m_ln, s_ln, alpha, mu_min = unpack(th)
        # pdf
        ln_pdf = lognormal_pdf(mu, m_ln, s_ln)
        pa_pdf = pareto_pdf(mu, alpha, mu_min)
        mix_pdf = (1.0 - w)*ln_pdf + w*pa_pdf
        lam = N * bw * mix_pdf   # 期望 count
        # Poisson NLL（忽略常数 log(cnt!)，防数值问题加下界）
        lam = np.clip(lam, 1e-300, None)
        return float(np.sum(lam - cnt*np.log(lam)))

    th0 = pack(w0, m0, s0, a0, mu_min0)
    res = minimize(nll, th0, method="L-BFGS-B")
    if not res.success:
        return None
    w, m_ln, s_ln, alpha, mu_min = unpack(res.x)

    # 生成与直方图对齐的三条“计数曲线”
    ln_pdf = lognormal_pdf(mu, m_ln, s_ln)
    pa_pdf = pareto_pdf(mu, alpha, mu_min)
    mix_pdf = (1.0 - w)*ln_pdf + w*pa_pdf
    y_ln  = N * bw * ((1.0 - w)*ln_pdf)
    y_pa  = N * bw * (w*pa_pdf)
    y_mix = N * bw * mix_pdf

    return dict(
        curves={"x": mu, "ln": y_ln, "pareto": y_pa, "mix": y_mix},
        params={"w": float(w), "m_ln": float(m_ln), "s_ln": float(s_ln),
                "alpha": float(alpha), "mu_min": float(mu_min)}
    )
