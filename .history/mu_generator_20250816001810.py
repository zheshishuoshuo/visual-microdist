"""Utilities to fit continuous p(mu|kappa,gamma,s) distributions.

This module implements a simple pipeline described in the repository
manual for building a continuous generator from histogram data.  It can
load cached histograms, fit mixture models, build an interpolator for
parameters, evaluate the analytic PDF/CDF and draw random samples.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import interpolate, optimize, special


# ---------------------------------------------------------------------------
# 1. Histogram utilities
# ---------------------------------------------------------------------------

def load_interpolated(cache_dir: str = "data_cache", dL: float = 0.01) -> Tuple[np.ndarray, List[dict]]:
    """Load cached histograms and interpolate them onto a common grid.

    Parameters
    ----------
    cache_dir:
        Directory where ``data_prep.py`` wrote its output.
    dL:
        Bin width in ``log(mu)``.  The default ``0.01`` matches the raw
        histogram spacing.

    Returns
    -------
    grid:
        1-D array of ``log(mu)`` grid points.
    items:
        List of dictionaries, each containing ``rid``, the interpolated
        ``cnt`` array and the total count ``N``.
    """

    hist_dir = os.path.join(cache_dir, "hists")
    if not os.path.isdir(hist_dir):
        raise FileNotFoundError(f"missing histogram cache: {hist_dir}")

    manifests: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for fname in sorted(os.listdir(hist_dir)):
        if not fname.endswith(".npz"):
            continue
        rid = os.path.splitext(fname)[0]
        z = np.load(os.path.join(hist_dir, fname))
        if "logmu_mid" not in z or "cnt_log" not in z:
            continue
        manifests.append((rid, z["logmu_mid"].astype(float), z["cnt_log"].astype(float)))

    if not manifests:
        raise RuntimeError("no histograms found in cache")

    # global grid
    y_all = np.concatenate([m[1] for m in manifests])
    y_min, y_max = float(y_all.min()), float(y_all.max())
    grid = np.arange(y_min, y_max + dL * 0.5, dL)

    items: List[dict] = []
    for rid, y, cnt in manifests:
        cnt_interp = np.interp(grid, y, cnt, left=0.0, right=0.0)
        items.append({
            "rid": rid,
            "cnt": cnt_interp,
            "N": float(cnt_interp.sum())
        })
    return grid, items


# ---------------------------------------------------------------------------
# 2. Analytic distribution
# ---------------------------------------------------------------------------

def _lognormal_pdf(mu: np.ndarray, m: float, s: float) -> np.ndarray:
    mu = np.asarray(mu, float)
    denom = np.maximum(mu * s * np.sqrt(2 * np.pi), 1e-12)
    z = (np.log(mu) - m) / np.maximum(s, 1e-12)
    return np.exp(-0.5 * z ** 2) / denom


def _lognormal_cdf(mu: np.ndarray, m: float, s: float) -> np.ndarray:
    z = (np.log(mu) - m) / np.maximum(s, 1e-12)
    return 0.5 * (1.0 + special.erf(z / np.sqrt(2.0)))


def _tail_pdf(mu: np.ndarray, alpha: float, mu0: float) -> np.ndarray:
    const = mu0 ** (1 - alpha) * special.gamma(alpha - 1)
    return mu ** (-alpha) * np.exp(-mu0 / mu) / const


def _tail_cdf(mu: np.ndarray, alpha: float, mu0: float) -> np.ndarray:
    # Regularised upper incomplete gamma Γ(a,x)/Γ(a)
    return special.gammaincc(alpha - 1, mu0 / mu)


def analytic_p_mu(mu: np.ndarray, psi: Iterable[float]) -> np.ndarray:
    """Evaluate the analytic PDF p(mu; psi)."""
    A, m, s, alpha, mu0 = psi
    mu = np.asarray(mu, float)
    return A * _lognormal_pdf(mu, m, s) + (1 - A) * _tail_pdf(mu, alpha, mu0)


def analytic_cdf_mu(mu: np.ndarray, psi: Iterable[float]) -> np.ndarray:
    """CDF corresponding to :func:`analytic_p_mu`."""
    A, m, s, alpha, mu0 = psi
    mu = np.asarray(mu, float)
    return A * _lognormal_cdf(mu, m, s) + (1 - A) * _tail_cdf(mu, alpha, mu0)


# ---------------------------------------------------------------------------
# 3. Single histogram fit
# ---------------------------------------------------------------------------




import numpy as np
from scipy import optimize

def _binwidth_mu_from_logy(y: np.ndarray) -> np.ndarray:
    """等距 log(mu) 网格对应的 Δμ（精确差分）。"""
    dL = float(np.median(np.diff(y)))
    return np.exp(y + 0.5 * dL) - np.exp(y - 0.5 * dL)

def _trim_by_mass(y: np.ndarray, cnt: np.ndarray, mass_lo=1e-4, mass_hi=1e-3):
    """按累计质量裁掉两侧极端稀疏端（默认保留 ~99.89% 质量）。"""
    N = cnt.sum()
    if N <= 0:
        return y, cnt
    q = cnt / N
    c = np.cumsum(q)
    i0 = int(np.searchsorted(c, mass_lo))
    i1 = int(np.searchsorted(c, 1.0 - mass_hi))
    i0 = max(0, min(i0, len(y)-4))
    i1 = max(i0+3, min(i1, len(y)))
    return y[i0:i1], cnt[i0:i1]

def fit_single(y: np.ndarray,
               cnt: np.ndarray,
               N: float | None = None,
               sigma_min: float = 0.15,
               use_two_peaks: bool = False,
               cdf_weight: float = 0.1,
               alpha_prior_w: float = 0.01) -> np.ndarray:
    """
    鲁棒单例拟合（Poisson NLL + 轻微CDF约束 + α≈3先验）。
    - sigma_min: 防止尖峰
    - use_two_peaks: True 时使用 双 lognormal + 尾巴
    - cdf_weight: 追加 Cramér–von Mises 残差的权重（改善尾部/形状）
    - alpha_prior_w: 尾部 α 的弱正则权重（把 α 轻推向 3）
    返回 ψ：
      单峰: (A, m, s, alpha, mu0)
      双峰: (A1, m1, s1, A2, m2, s2, alpha, mu0)  其中 A_tail = 1 - A1 - A2
    """
    y = np.asarray(y, float)
    cnt = np.asarray(cnt, float)
    mask = np.isfinite(y) & np.isfinite(cnt) & (cnt >= 0)
    y, cnt = y[mask], cnt[mask]
    if y.size < 6 or cnt.sum() <= 0:
        raise ValueError("insufficient data to fit")

    # 裁剪极端两端，避免大量零计数把模型“挤尖”
    y, cnt = _trim_by_mass(y, cnt, mass_lo=5e-4, mass_hi=5e-4)
    if y.size < 6:
        raise ValueError("too few bins after trimming")

    if N is None:
        N = float(cnt.sum())

    mu  = np.exp(y)
    dmu = _binwidth_mu_from_logy(y)

    # ====== 模型：单峰或双峰 ======
    def pdf_single(theta):
        A, m, s, alpha, mu0 = theta
        s     = max(s, sigma_min)
        alpha = max(alpha, 1.01)
        mu0   = max(mu0, 1e-12)
        # lognormal
        denom = np.maximum(mu * s * np.sqrt(2*np.pi), 1e-12)
        z = (np.log(mu) - m) / s
        ln = np.exp(-0.5*z*z) / denom
        # tail ~ mu^{-alpha} exp(-mu0/mu) / Z
        Z = (mu0**(1-alpha)) * np.math.gamma(alpha-1.0)
        tail = np.power(mu, -alpha) * np.exp(-mu0/mu) / np.maximum(Z, 1e-300)
        pdf = A*ln + (1.0 - A)*tail
        # 归一化保护
        Zp = np.trapz(pdf, mu);  pdf = pdf / np.maximum(Zp, 1e-300)
        return pdf

    def pdf_two(theta):
        # params: (A1, m1, s1, A2, m2, s2, alpha, mu0) with A_tail = 1-A1-A2
        A1, m1, s1, A2, m2, s2, alpha, mu0 = theta
        s1 = max(s1, sigma_min); s2 = max(s2, sigma_min)
        alpha = max(alpha, 1.01); mu0 = max(mu0, 1e-12)
        A1 = np.clip(A1, 0.0, 1.0); A2 = np.clip(A2, 0.0, 1.0)
        A_tail = max(0.0, 1.0 - A1 - A2)

        def ln_part(m, s):
            denom = np.maximum(mu * s * np.sqrt(2*np.pi), 1e-12)
            z = (np.log(mu) - m) / s
            return np.exp(-0.5*z*z) / denom

        ln1 = ln_part(m1, s1)
        ln2 = ln_part(m2, s2)
        Z = (mu0**(1-alpha)) * np.math.gamma(alpha-1.0)
        tail = np.power(mu, -alpha) * np.exp(-mu0/mu) / np.maximum(Z, 1e-300)
        pdf = A1*ln1 + A2*ln2 + A_tail*tail
        Zp = np.trapz(pdf, mu);  pdf = pdf / np.maximum(Zp, 1e-300)
        return pdf

    pdf = pdf_two if use_two_peaks else pdf_single

    # ====== 损失：Poisson NLL + 轻微 CDF 残差 + α 先验 ======
    def objective(theta):
        p = pdf(theta)
        lam = np.clip(N * p * dmu, 1e-30, None)
        # Poisson NLL（忽略常数 log n!）
        nll = np.sum(lam - cnt * np.log(lam))

        # CDF 残差（缓和形状/尾部偏差）
        Fm = np.cumsum(p * dmu); Fm /= max(Fm[-1], 1e-300)
        Fd = np.cumsum(cnt);     Fd /= max(Fd[-1], 1e-300)
        cvm = np.mean((Fm - Fd)**2)

        # α≈3 的弱先验（源越大可加大权重，这里统一系数）
        if use_two_peaks:
            alpha = theta[6]
        else:
            alpha = theta[3]
        prior = (alpha - 3.0)**2

        return nll + cdf_weight * cvm + alpha_prior_w * prior

    # ====== 初值 ======
    # 中位/四分位估计 m, s；尾部初值近似用 y-范围 & 最小 μ
    med = float(np.median(y))
    i_max = int(np.argmax(cnt))
    m0 = float(y[i_max])
    s0 = max(float(np.std(y, ddof=1))*0.5, sigma_min)
    alpha0 = 3.0
    mu0_0 = float(np.exp(y.min()))

    if use_two_peaks:
        # 两个峰：在主峰两侧给两个初值，权重均分
        x0 = np.array([0.4, m0 - 0.4, s0, 0.4, m0 + 0.4, s0, alpha0, mu0_0])
        bounds_lo = [0.0, y.min()-5, sigma_min, 0.0, y.min()-5, sigma_min, 1.01, 0.0]
        bounds_hi = [1.0, y.max()+5, 5.0,       1.0, y.max()+5, 5.0,       10.0, np.exp(y.max())]
    else:
        x0 = np.array([0.7, m0, s0, alpha0, mu0_0])
        bounds_lo = [0.0, y.min()-5, sigma_min, 1.01, 0.0]
        bounds_hi = [1.0, y.max()+5, 5.0,       10.0, np.exp(y.max())]

    res = optimize.minimize(objective, x0=x0, bounds=list(zip(bounds_lo, bounds_hi)),
                            method="L-BFGS-B", options=dict(maxiter=500))
    return res.x



# def fit_single(y: np.ndarray, cnt: np.ndarray, N: float | None = None) -> np.ndarray:
#     """Fit parameters ``psi`` for a single histogram.

#     Parameters
#     ----------
#     y : array_like
#         Grid of ``log(mu)`` midpoints.
#     cnt : array_like
#         Histogram counts on the same grid.
#     N : float, optional
#         Total count.  If ``None`` it will be ``cnt.sum()``.
#     """

#     y = np.asarray(y, float)
#     cnt = np.asarray(cnt, float)
#     mask = np.isfinite(y) & np.isfinite(cnt) & (cnt >= 0)
#     y, cnt = y[mask], cnt[mask]
#     if y.size < 4 or cnt.sum() <= 0:
#         raise ValueError("insufficient data to fit")

#     if N is None:
#         N = float(cnt.sum())
#     dL = np.median(np.diff(y))
#     mu = np.exp(y)
#     dmu = np.exp(y + 0.5 * dL) - np.exp(y - 0.5 * dL)
#     weights = 1.0 / np.sqrt(cnt + 1.0)

#     def residual(params: np.ndarray) -> np.ndarray:
#         m = N * analytic_p_mu(mu, params) * dmu
#         return (m - cnt) * weights

#     x0 = np.array([0.8, np.median(y), 0.5, 3.0, np.exp(y.min())])
#     bounds = ([0.0, y.min() - 5.0, 0.05, 1.01, 0.0],
#               [1.0, y.max() + 5.0, 5.0, 10.0, np.exp(y.max())])
#     res = optimize.least_squares(residual, x0, bounds=bounds, loss="soft_l1")
#     return res.x


# ---------------------------------------------------------------------------
# 4. psi-(kappa,gamma,s) mapping
# ---------------------------------------------------------------------------

@dataclass
class PsiModel:
    """Callable interpolator mapping (kappa, gamma, s) to ``psi``."""

    rbf: interpolate.RBFInterpolator

    def __call__(self, kappa: float, gamma: float, s: float) -> np.ndarray:
        pts = np.column_stack([
            np.atleast_1d(kappa), np.atleast_1d(gamma), np.atleast_1d(s)
        ])
        out = self.rbf(pts)
        return out[0] if out.shape[0] == 1 else out


def build_psi_model(df: pd.DataFrame, psi_list: List[np.ndarray]) -> PsiModel:
    """Build an RBF interpolator from fitted ``psi`` values."""
    pts = df[["kappa", "gamma", "s"]].to_numpy()
    vals = np.asarray(psi_list)
    rbf = interpolate.RBFInterpolator(pts, vals)
    return PsiModel(rbf)


# ---------------------------------------------------------------------------
# 5. High-level evaluation and sampling
# ---------------------------------------------------------------------------

def p_mu_given_eta(mu: np.ndarray, kappa: float, gamma: float, s: float, model: PsiModel) -> np.ndarray:
    """Evaluate p(mu | kappa, gamma, s)."""
    psi = model(kappa, gamma, s)
    return analytic_p_mu(mu, psi)


def sample_mu(kappa: float, gamma: float, s: float, model: PsiModel, size: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw samples of ``mu`` given ``(kappa, gamma, s)`` using inverse transform."""
    if rng is None:
        rng = np.random.default_rng()
    psi = model(kappa, gamma, s)
    A, m, sig, alpha, mu0 = psi
    n_ln = rng.binomial(size, A)
    out: List[np.ndarray] = []
    if n_ln:
        out.append(rng.lognormal(mean=m, sigma=sig, size=n_ln))
    n_tail = size - n_ln
    if n_tail:
        t = rng.gamma(shape=alpha - 1.0, scale=1.0, size=n_tail)
        out.append(mu0 / t)
    return np.concatenate(out) if out else np.empty(0)


__all__ = [
    "load_interpolated",
    "analytic_p_mu",
    "analytic_cdf_mu",
    "fit_single",
    "build_psi_model",
    "p_mu_given_eta",
    "sample_mu",
    "PsiModel",
]


def main(argv: List[str] | None = None) -> None:
    """Command line interface for sampling ``mu`` values.

    This utility builds a small ``PsiModel`` from cached histograms and
    draws random samples for user supplied ``(kappa, gamma, s)``.
    ``--limit`` can be used to restrict the number of histograms fitted in
    order to keep the example lightweight.
    """

    parser = argparse.ArgumentParser(description="Sample from p(mu | kappa, gamma, s)")
    parser.add_argument("--kappa", type=float, required=True, help="kappa value")
    parser.add_argument("--gamma", type=float, required=True, help="gamma value")
    parser.add_argument("--s", type=float, required=True, help="s value")
    parser.add_argument("--size", type=int, default=1, help="number of samples to draw")
    parser.add_argument(
        "--cache-dir",
        default="data_cache",
        help="directory containing histogram cache and samples.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="number of histograms to fit when building the model",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args(argv)

    grid, items = load_interpolated(args.cache_dir)
    df = pd.read_csv(os.path.join(args.cache_dir, "samples.csv")).set_index("rid")

    psi_list: List[np.ndarray] = []
    params: List[np.ndarray] = []
    for item in items[: args.limit]:
        rid = item["rid"]
        if rid not in df.index:
            continue
        psi_list.append(fit_single(grid, item["cnt"], item["N"]))
        params.append(df.loc[rid, ["kappa", "gamma", "s"]].to_numpy())

    if not psi_list:
        raise RuntimeError("no histograms fitted; check cache directory")

    model_df = pd.DataFrame(params, columns=["kappa", "gamma", "s"])
    model = build_psi_model(model_df, psi_list)
    rng = np.random.default_rng(args.seed)
    samples = sample_mu(args.kappa, args.gamma, args.s, model, args.size, rng)
    for mu in samples:
        print(mu)


if __name__ == "__main__":
    main()
