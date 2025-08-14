# fit_utils.py
import numpy as np
from functools import lru_cache
from scipy import special
from scipy.optimize import minimize


EPS = 1e-12
EXP_CLIP = 50.0



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
    if not np.any(mask):
        return out
    mu_safe = np.clip(mu[mask], EPS, None)
    s_ln_safe = np.maximum(s_ln, EPS)
    z = (np.log(mu_safe) - m_ln) / s_ln_safe
    z = np.clip(z, -EXP_CLIP, EXP_CLIP)
    denom = np.maximum(mu_safe * s_ln_safe * np.sqrt(2 * np.pi), EPS)
    out[mask] = np.exp(-0.5 * z * z) / denom
    return out

def lognormal_cdf(mu, m_ln, s_ln):
    mu = np.asarray(mu, float)
    mask = (mu > 0) & np.isfinite(mu)
    out = np.zeros_like(mu, dtype=float)
    if not np.any(mask):
        return out
    mu_safe = np.clip(mu[mask], EPS, None)
    s_ln_safe = np.maximum(s_ln, EPS)
    z = (np.log(mu_safe) - m_ln) / s_ln_safe
    z = np.clip(z, -EXP_CLIP, EXP_CLIP)
    out[mask] = _norm_cdf(z)
    return out

def pareto_pdf(mu, alpha, mu0):
    """标准 Pareto（下限 mu0），alpha>1 保证可积"""
    mu = np.asarray(mu, float)
    out = np.zeros_like(mu, dtype=float)
    m = (mu >= mu0) & np.isfinite(mu)
    if alpha <= 1 or not np.any(m):
        return out
    mu_safe = np.clip(mu[m], EPS, None)
    out[m] = (alpha - 1.0) * (mu0 ** (alpha - 1.0)) * (mu_safe ** (-alpha))
    return out

# ========= 拟合：Lognormal（μ≤μ0） + Pareto（μ≥μ0） =========

def fit_lognorm_bulk(mu, cnt, mu0):
    """对 μ≤μ0 的主体做加权 MLE：返回 (m_ln, s_ln, mass_bulk, Z_bulk)
       mass_bulk = 主体权重（计数占比），Z_bulk = 截断归一化因子 Φ=P(μ≤μ0)
    """
    mu = np.asarray(mu, float); cnt = np.asarray(cnt, float)
    ok = np.isfinite(mu) & np.isfinite(cnt)
    m = (mu > 0) & (mu <= mu0) & (cnt > 0) & ok
    if m.sum() < 3:
        return None
    w = cnt[m]
    x = np.log(np.clip(mu[m], EPS, None))
    wsum = np.maximum(w.sum(), EPS)
    m_ln = (w * x).sum() / wsum
    s2 = (w * (x - m_ln) ** 2).sum() / wsum
    s_ln = np.sqrt(max(s2, EPS))
    # 截断因子（Φ = P[μ<=μ0]）
    Z = float(lognormal_cdf(np.array([mu0]), m_ln, s_ln)[0])
    total = cnt[(cnt > 0) & ok & (mu > 0)].sum()
    mass_bulk = float(w.sum() / np.maximum(total, EPS))
    return m_ln, s_ln, mass_bulk, Z

def fit_pareto_tail(mu, cnt, mu0):
    """对 μ≥μ0 的尾部用加权 Hill 估计：alpha = 1 + W / sum(w*ln(mu/mu0))"""
    mu = np.asarray(mu, float); cnt = np.asarray(cnt, float)
    ok = np.isfinite(mu) & np.isfinite(cnt)
    m = (mu >= mu0) & (cnt > 0) & ok
    if m.sum() < 2:
        return None
    w = cnt[m]
    log_term = np.log(np.clip(mu[m] / mu0, EPS, None))
    denom = (w * log_term).sum()
    if denom <= 0:
        return None
    alpha = 1.0 + w.sum() / np.maximum(denom, EPS)
    total = cnt[(cnt > 0) & ok & (mu > 0)].sum()
    mass_tail = float(w.sum() / np.maximum(total, EPS))
    return alpha, mass_tail

def build_piecewise_pdf(mu_grid, mu0, m_ln, s_ln, alpha, mass_bulk, mass_tail, Z_bulk):
    """返回分段混合 pdf： μ≤μ0 用截断LN， μ≥μ0 用Pareto；保证总质量=1"""
    pdf = np.zeros_like(mu_grid, dtype=float)
    left = mu_grid <= mu0
    right = ~left
    if left.any():
        f = lognormal_pdf(mu_grid[left], m_ln, s_ln)
        f = f / np.maximum(Z_bulk, EPS)
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



def fit_and_predict(mu, cnt, pct=95.0, grid_points=2000, bin_width=None):
    mu = np.asarray(mu, float)
    cnt = np.asarray(cnt, float)
    ok = np.isfinite(mu) & np.isfinite(cnt) & (mu > 0) & (cnt > 0)
    mu, cnt = mu[ok], cnt[ok]
    if mu.size < 8 or cnt.sum() <= 0:
        return None

    if bin_width is None:
        bin_width = np.median(np.diff(mu))

    mu0 = weighted_quantile(mu, cnt, pct / 100.0)
    if not np.isfinite(mu0):
        return None

    bulk = fit_lognorm_bulk(mu, cnt, mu0)
    tail = fit_pareto_tail(mu, cnt, mu0)
    if bulk is None or tail is None:
        return None

    m_ln, s_ln, mass_bulk, Z = bulk
    alpha, mass_tail = tail

    N = np.maximum(cnt.sum(), EPS)
    mu_grid = np.linspace(mu.min(), mu.max(), grid_points)
    pdf_mix = build_piecewise_pdf(mu_grid, mu0, m_ln, s_ln, alpha, mass_tail, mass_tail, Z)
    y_mix = predict_counts_curve(mu_grid, pdf_mix, N, bin_width)

    pdf_bulk = np.zeros_like(mu_grid)
    pdf_tail = np.zeros_like(mu_grid)
    left = mu_grid <= mu0
    right = ~left
    if left.any():
        f = lognormal_pdf(mu_grid[left], m_ln, s_ln) / np.maximum(Z, EPS)
        pdf_bulk[left] = (1.0 - mass_tail) * f
    if right.any():
        pdf_tail[right] = mass_tail * pareto_pdf(mu_grid[right], alpha, mu0)

    y_bulk = predict_counts_curve(mu_grid, pdf_bulk, N, bin_width)
    y_tail = predict_counts_curve(mu_grid, pdf_tail, N, bin_width)

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
    mu = np.asarray(mu, float)
    cnt = np.asarray(cnt, float)
    bw = np.asarray(bin_widths, float)
    ok = (
        np.isfinite(mu)
        & np.isfinite(cnt)
        & np.isfinite(bw)
        & (bw > 0)
        & (mu > 0)
        & (cnt > 0)
    )
    mu, cnt, bw = mu[ok], cnt[ok], bw[ok]
    if mu.size < 8 or cnt.sum() <= 0:
        return None

    N = float(np.maximum(cnt.sum(), EPS))
    # μ_min 用加权分位数初始化（也可让用户通过滑条控制）
    mu_min0 = max(1e-6, weighted_quantile(mu, cnt, pct_for_mu_min/100.0))
    # 初值：先用未截断 LN 的矩来初始化 m, s；w 初始化为尾部质量占比；alpha 给个温和初值
    x = np.log(np.clip(mu, EPS, None))
    wts = cnt
    wsum = np.maximum(wts.sum(), EPS)
    m0 = (wts * x).sum() / wsum
    s0 = np.sqrt(max((wts * (x - m0) ** 2).sum() / wsum, 1e-3))
    w0 = min(0.4, float(cnt[mu >= mu_min0].sum() / N))   # 初始尾部权重
    a0 = 3.0

    # 变量向量：theta = [logit(w), m_ln, log(s_ln), log(alpha-1), log(mu_min - mu_lo)]
    mu_lo = float(np.min(mu))

    def pack(w, m_ln, s_ln, alpha, mu_min):
        w = np.clip(w, EPS, 1 - EPS)
        z0 = np.log(w / np.maximum(1 - w, EPS))
        return np.array(
            [
                z0,
                m_ln,
                np.log(np.maximum(s_ln, EPS)),
                np.log(np.maximum(alpha - 1.0, EPS)),
                np.log(np.maximum(mu_min - mu_lo, 1e-9)),
            ],
            float,
        )

    def unpack(th):
        z0, m_ln, ls, la1, lmm = th
        z0 = np.clip(z0, -EXP_CLIP, EXP_CLIP)
        w = 1.0 / (1.0 + np.exp(-z0))       # (0,1)
        w = np.clip(w, EPS, 1 - EPS)
        s_ln = np.exp(np.clip(ls, -EXP_CLIP, EXP_CLIP))
        s_ln = np.maximum(s_ln, EPS)
        alpha = 1.0 + np.exp(np.clip(la1, -EXP_CLIP, EXP_CLIP))
        mu_min = mu_lo + np.exp(np.clip(lmm, -EXP_CLIP, EXP_CLIP))
        return w, m_ln, s_ln, alpha, mu_min

    def nll(th):
        w, m_ln, s_ln, alpha, mu_min = unpack(th)
        ln_pdf = lognormal_pdf(mu, m_ln, s_ln)
        pa_pdf = pareto_pdf(mu, alpha, mu_min)
        mix_pdf = (1.0 - w) * ln_pdf + w * pa_pdf
        lam = N * bw * mix_pdf
        lam = np.clip(lam, 1e-300, None)
        return float(np.sum(lam - cnt * np.log(lam)))

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


# ========= 可扩展拟合框架 =========

FITTERS = {}


def register_fitter(name):
    """装饰器：注册一个拟合方法"""
    def _decorator(cls):
        FITTERS[name] = cls
        return cls
    return _decorator


class BaseFitModel:
    """拟合模型基类：子类需实现 _default_init、_default_bounds、_pdf"""

    param_names = []

    def __init__(self, init_params=None, bounds=None, fixed=None):
        self.init_params = init_params or {}
        self.bounds = bounds or {}
        self.fixed = set(fixed or [])

    # ----- 钩子：默认初值与范围 -----
    def _default_init(self, mu, cnt, bw):
        return {}

    def _default_bounds(self, mu, cnt, bw):
        return {name: (None, None) for name in self.param_names}

    # ----- 子类需提供的 pdf -----
    def _pdf(self, mu, params):
        raise NotImplementedError

    # ----- NLL 与曲线生成 -----
    def _nll(self, mu, cnt, bw, params):
        N = np.maximum(cnt.sum(), EPS)
        pdf = self._pdf(mu, params)
        lam = N * bw * pdf
        lam = np.clip(lam, 1e-300, None)
        return float(np.sum(lam - cnt * np.log(lam)))

    def _predict_curves(self, mu, cnt, bw, params):
        N = np.maximum(cnt.sum(), EPS)
        pdf = self._pdf(mu, params)
        y = N * bw * pdf
        return {"x": mu, "fit": y}

    # ----- 主入口 -----
    def fit(self, mu, cnt, bw):
        mu = np.asarray(mu, float)
        cnt = np.asarray(cnt, float)
        bw = np.asarray(bw, float)
        ok = (
            np.isfinite(mu)
            & np.isfinite(cnt)
            & np.isfinite(bw)
            & (bw > 0)
            & (mu > 0)
            & (cnt > 0)
        )
        mu, cnt, bw = mu[ok], cnt[ok], bw[ok]
        if mu.size < 2 or cnt.sum() <= 0:
            return None

        init = {**self._default_init(mu, cnt, bw), **self.init_params}
        bounds = {**self._default_bounds(mu, cnt, bw), **self.bounds}

        free = [p for p in self.param_names if p not in self.fixed]
        if not free:
            params = {p: init[p] for p in self.param_names}
            curves = self._predict_curves(mu, cnt, bw, params)
            return {"curves": curves, "params": {k: float(v) for k, v in params.items()}}

        x0 = np.array([init[p] for p in free], float)
        bnds = [bounds.get(p, (None, None)) for p in free]

        def obj(x):
            params = dict(zip(free, x))
            for p in self.fixed:
                params[p] = init[p]
            return self._nll(mu, cnt, bw, params)

        res = minimize(obj, x0, method="L-BFGS-B", bounds=bnds)
        if not res.success:
            return None
        params = dict(zip(free, res.x))
        for p in self.fixed:
            params[p] = init[p]

        curves = self._predict_curves(mu, cnt, bw, params)
        return {"curves": curves, "params": {k: float(v) for k, v in params.items()}}


@register_fitter("lognormal")
class LogNormalModel(BaseFitModel):
    param_names = ["m_ln", "s_ln"]

    def _default_init(self, mu, cnt, bw):
        x = np.log(np.clip(mu, EPS, None))
        w = cnt
        wsum = np.maximum(w.sum(), EPS)
        m0 = (w * x).sum() / wsum
        s0 = np.sqrt(max((w * (x - m0) ** 2).sum() / wsum, 1e-3))
        return {"m_ln": m0, "s_ln": s0}

    def _default_bounds(self, mu, cnt, bw):
        return {"m_ln": (None, None), "s_ln": (1e-6, None)}

    def _pdf(self, mu, params):
        return lognormal_pdf(mu, params["m_ln"], params["s_ln"])


@register_fitter("lognorm_pareto")
@register_fitter("lognormal+pareto")
class LogNormParetoModel(BaseFitModel):
    param_names = ["w", "m_ln", "s_ln", "alpha", "mu_min"]

    def __init__(self, init_params=None, bounds=None, fixed=None, pct_for_mu_min=95.0):
        super().__init__(init_params, bounds, fixed)
        self.pct_for_mu_min = pct_for_mu_min

    def _default_init(self, mu, cnt, bw):
        N = np.maximum(cnt.sum(), EPS)
        mu_min0 = max(1e-6, weighted_quantile(mu, cnt, self.pct_for_mu_min / 100.0))
        x = np.log(np.clip(mu, EPS, None))
        wts = cnt
        wsum = np.maximum(wts.sum(), EPS)
        m0 = (wts * x).sum() / wsum
        s0 = np.sqrt(max((wts * (x - m0) ** 2).sum() / wsum, 1e-3))
        w0 = min(0.4, float(cnt[mu >= mu_min0].sum() / N))
        a0 = 3.0
        return {"w": w0, "m_ln": m0, "s_ln": s0, "alpha": a0, "mu_min": mu_min0}

    def _default_bounds(self, mu, cnt, bw):
        mu_lo = float(np.min(mu))
        return {
            "w": (1e-6, 1 - 1e-6),
            "m_ln": (None, None),
            "s_ln": (1e-6, None),
            "alpha": (1 + 1e-6, None),
            "mu_min": (mu_lo, None),
        }

    def _pdf(self, mu, params):
        ln_pdf = lognormal_pdf(mu, params["m_ln"], params["s_ln"])
        pa_pdf = pareto_pdf(mu, params["alpha"], params["mu_min"])
        return (1.0 - params["w"]) * ln_pdf + params["w"] * pa_pdf

    def _predict_curves(self, mu, cnt, bw, params):
        N = np.maximum(cnt.sum(), EPS)
        ln_pdf = lognormal_pdf(mu, params["m_ln"], params["s_ln"])
        pa_pdf = pareto_pdf(mu, params["alpha"], params["mu_min"])
        y_ln = N * bw * ((1.0 - params["w"]) * ln_pdf)
        y_pa = N * bw * (params["w"] * pa_pdf)
        y_mix = y_ln + y_pa
        return {"x": mu, "ln": y_ln, "pareto": y_pa, "mix": y_mix}


@register_fitter("exponential")
class ExponentialModel(BaseFitModel):
    """单参数指数分布: pdf = λ exp(-λ μ), μ≥0"""

    param_names = ["lam"]

    def _default_init(self, mu, cnt, bw):
        mu_safe = np.clip(mu, EPS, None)
        w = cnt
        denom = np.maximum((w * mu_safe).sum(), EPS)
        lam0 = w.sum() / denom
        return {"lam": lam0}

    def _default_bounds(self, mu, cnt, bw):
        return {"lam": (1e-6, None)}

    def _pdf(self, mu, params):
        mu = np.asarray(mu, float)
        lam = float(params["lam"])
        out = np.zeros_like(mu, float)
        m = (mu >= 0)
        out[m] = lam * np.exp(-lam * mu[m])
        return out


def fit_histogram(mu, cnt, bin_widths, method="lognormal", init_params=None,
                  bounds=None, fixed=None, **method_kwargs):
    """高层接口：按指定 method 拟合直方图

    Parameters
    ----------
    mu : array
        直方图中心
    cnt : array
        直方图计数
    bin_widths : array or float
        每个 bin 的宽度
    method : str
        拟合方法名称，例如 'lognormal'、'lognorm_pareto'
    init_params/bounds/fixed : dict
        运行时覆盖初值、参数范围或固定某些参数
    method_kwargs : dict
        传给具体拟合类的附加参数
    """

    bw = bin_widths
    if np.isscalar(bw):
        bw = np.full_like(mu, float(bw))

    cls = FITTERS.get(method)
    if cls is None:
        raise ValueError(f"Unknown fit method: {method}")

    fitter = cls(init_params=init_params, bounds=bounds, fixed=fixed, **method_kwargs)
    return fitter.fit(mu, cnt, np.asarray(bw, float))

