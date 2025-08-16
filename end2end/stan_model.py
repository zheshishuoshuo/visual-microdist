"""Stan-based conditional analytic distribution fitting.

This module provides utilities to prepare data for the Stan model
``pmu_conditional.stan`` and to fit it using :mod:`cmdstanpy`.  It reuses the
histogram loading and basis feature helpers from :mod:`end2end.model` but
delegates optimisation to Stan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from cmdstanpy import CmdStanModel, CmdStanMLE
from scipy.special import gamma

from .model import basis_features, load_dataset


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_stan_data(
    cache_dir: str,
    basis_fn: Callable[[np.ndarray], np.ndarray] = basis_features,
    *,
    limit: int | None = None,
    K: int = 2,
    sigma_min: float = 0.05,
    lambda_ent: float = 1e-3,
    lambda_tail: float = 1e-3,
    eq_weight: int = 1,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load cached histograms and assemble a Stan ``data`` dictionary.

    Parameters
    ----------
    cache_dir:
        Directory containing the preprocessed histograms produced by
        ``data_prep.py``.
    basis_fn:
        Callable mapping ``η=(κ, γ, s)`` to feature vectors.
    limit:
        Optional maximum number of histogram files to load.
    K:
        Number of log-normal mixture components in the Stan model.
    sigma_min, lambda_ent, lambda_tail, eq_weight:
        Hyper-parameters passed straight to Stan.

    Returns
    -------
    stan_data, aux
        ``stan_data`` is a dictionary compatible with ``pmu_conditional.stan``;
        ``aux`` contains feature normalisation statistics that must be supplied
        when predicting new ``η`` values.
    """

    data = load_dataset(cache_dir, limit=limit)
    if not data:
        raise RuntimeError("no data loaded; ensure cache directory is correct")

    # check common log-mu grid
    y0 = data[0]["y"]
    if not all(np.allclose(d["y"], y0) for d in data):
        raise ValueError("all histograms must share the same log-mu grid")
    mu = np.exp(y0)
    dmu = np.exp(y0 + 0.5 * (y0[1] - y0[0])) - np.exp(y0 - 0.5 * (y0[1] - y0[0]))

    cnt = np.vstack([d["cnt"] for d in data]).astype(int)
    Ntot = np.array([d["N"] for d in data])

    feats = np.vstack([basis_fn(d["eta"]) for d in data])
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    B = (feats - mean) / (std + 1e-12)

    stan_data: Dict[str, np.ndarray] = {
        "I": cnt.shape[0],
        "J": cnt.shape[1],
        "mu": mu,
        "dmu": dmu,
        "cnt": cnt,
        "Ntot": Ntot,
        "P": B.shape[1],
        "B": B,
        "K": K,
        "sigma_min": sigma_min,
        "lambda_ent": lambda_ent,
        "lambda_tail": lambda_tail,
        "eq_weight": eq_weight,
    }
    aux = {"feat_mean": mean, "feat_std": std, "sigma_min": sigma_min, "K": K}
    return stan_data, aux


# ---------------------------------------------------------------------------
# Fitting and prediction
# ---------------------------------------------------------------------------

def fit_model(stan_data: Dict[str, np.ndarray], *, stan_file: str = "pmu_conditional.stan", **kwargs) -> Tuple[CmdStanModel, CmdStanMLE]:
    """Compile and fit the Stan model returning the model and MAP estimate."""

    model = CmdStanModel(stan_file=stan_file)
    fit = model.optimize(data=stan_data, **kwargs)
    return model, fit


def _lognormal_pdf(mu: np.ndarray, m: float, s: float) -> np.ndarray:
    return 1.0 / (mu * s * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((np.log(mu) - m) / s) ** 2)


def _tail_pdf(mu: np.ndarray, alpha: float, mu0: float) -> np.ndarray:
    return (mu0**alpha / gamma(alpha)) * mu ** (-(alpha + 1.0)) * np.exp(-mu0 / mu)


@dataclass
class StanState:
    fit: CmdStanMLE
    aux: Dict[str, np.ndarray]
    basis_fn: Callable[[np.ndarray], np.ndarray] = basis_features


def pdf_mu(eta: np.ndarray, mu: np.ndarray, state: StanState) -> np.ndarray:
    """Evaluate ``p(μ|η)`` using fitted Stan parameters."""

    fit = state.fit
    aux = state.aux
    K = int(aux["K"])
    sigma_min = float(aux["sigma_min"])
    phi = state.basis_fn(eta)
    phi = (phi - aux["feat_mean"]) / (aux["feat_std"] + 1e-12)

    beta_m1 = fit.stan_variable("beta_m1")
    if K > 1:
        beta_m_diff = fit.stan_variable("beta_m_diff")
    else:
        beta_m_diff = np.empty((beta_m1.shape[0], 0))
    beta_logsig = fit.stan_variable("beta_logsig")
    beta_w = fit.stan_variable("beta_w")
    beta_alpha = fit.stan_variable("beta_alpha")
    beta_mu0 = fit.stan_variable("beta_mu0")

    m1 = phi @ beta_m1
    means = [m1]
    if K > 1:
        deltas = phi @ beta_m_diff
        deltas = np.log1p(np.exp(deltas))
        for d in deltas:
            means.append(means[-1] + d)
    means = np.array(means)

    sig_raw = phi @ beta_logsig
    sigmas = sigma_min + np.log1p(np.exp(sig_raw))

    logits = phi @ beta_w
    w = np.exp(logits - logits.max())
    w = w / w.sum()

    alpha = 1.0 + np.log1p(np.exp(phi @ beta_alpha))
    mu0 = np.log1p(np.exp(phi @ beta_mu0))

    pdf = np.zeros_like(mu)
    for k in range(K):
        pdf += w[k] * _lognormal_pdf(mu, means[k], sigmas[k])
    pdf += w[K] * _tail_pdf(mu, alpha, mu0)
    return pdf

