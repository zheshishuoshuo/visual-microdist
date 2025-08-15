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

def fit_single(y: np.ndarray, cnt: np.ndarray, N: float | None = None) -> np.ndarray:
    """Fit parameters ``psi`` for a single histogram.

    Parameters
    ----------
    y : array_like
        Grid of ``log(mu)`` midpoints.
    cnt : array_like
        Histogram counts on the same grid.
    N : float, optional
        Total count.  If ``None`` it will be ``cnt.sum()``.
    """

    y = np.asarray(y, float)
    cnt = np.asarray(cnt, float)
    mask = np.isfinite(y) & np.isfinite(cnt) & (cnt >= 0)
    y, cnt = y[mask], cnt[mask]
    if y.size < 4 or cnt.sum() <= 0:
        raise ValueError("insufficient data to fit")

    if N is None:
        N = float(cnt.sum())
    dL = np.median(np.diff(y))
    mu = np.exp(y)
    dmu = np.exp(y + 0.5 * dL) - np.exp(y - 0.5 * dL)
    weights = 1.0 / np.sqrt(cnt + 1.0)

    def residual(params: np.ndarray) -> np.ndarray:
        m = N * analytic_p_mu(mu, params) * dmu
        return (m - cnt) * weights

    x0 = np.array([0.8, np.median(y), 0.5, 3.0, np.exp(y.min())])
    bounds = ([0.0, y.min() - 5.0, 0.05, 1.01, 0.0],
              [1.0, y.max() + 5.0, 5.0, 10.0, np.exp(y.max())])
    res = optimize.least_squares(residual, x0, bounds=bounds, loss="soft_l1")
    return res.x


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
