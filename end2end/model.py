from __future__ import annotations

"""End-to-end conditional analytic distribution model.

This module implements a small conditional density estimator for the
microlensing magnification distribution ``p(μ | η)`` described in the
user specification.  The distribution is represented as a mixture of
log-normal components with an optional Pareto tail.  All mixture
parameters are smooth functions of the lensing parameters
``η=(κ, γ, s)`` through a low degree polynomial basis.

The implementation focuses on robustness and ease of use so the code can
serve as a starting point for further experimentation.  Gradients are
computed with :mod:`autograd`, and optimisation relies on
:func:`scipy.optimize.minimize`.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List

import autograd.numpy as anp
from autograd import value_and_grad
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def basis_features(eta: anp.ndarray) -> anp.ndarray:
    """Compute polynomial features ``φ(η)``.

    Parameters
    ----------
    eta:
        Array-like of shape ``(3,)`` containing ``(κ, γ, s)``.

    Returns
    -------
    ndarray of shape ``(B,)`` containing polynomial features up to second
    order as well as additional physically motivated quantities.
    """

    kappa, gamma, s = eta
    macro_denom = (1.0 - kappa) ** 2 - gamma ** 2
    mu_macro = 1.0 / macro_denom
    delta = anp.abs(macro_denom)
    parity = anp.sign(macro_denom)

    # Second order polynomial basis
    feats = anp.array(
        [
            1.0,
            kappa,
            gamma,
            s,
            kappa ** 2,
            kappa * gamma,
            kappa * s,
            gamma ** 2,
            gamma * s,
            s ** 2,
            mu_macro,
            delta,
            parity,
        ]
    )
    return feats


# ---------------------------------------------------------------------------
# Mixture model and parameter mapping
# ---------------------------------------------------------------------------

@dataclass
class MixtureParams:
    """Container for decoded mixture parameters."""

    weights: anp.ndarray  # (K + 1,)
    means: anp.ndarray  # (K,)
    sigmas: anp.ndarray  # (K,)
    alpha: float
    mu0: float


class ConditionalMixtureModel:
    """Conditional mixture model with a Pareto tail."""

    def __init__(
        self,
        K: int = 2,
        basis_fn: Callable[[anp.ndarray], anp.ndarray] = basis_features,
        s_min: float = 0.2,
        eps: float = 1e-6,
    ) -> None:
        self.K = K
        self.basis_fn = basis_fn
        self.s_min = s_min
        self.eps = eps

        self.basis_dim = len(basis_fn(anp.array([0.0, 0.0, 0.0])))
        # number of parameter groups: logits(K+1), m1, Δm for k>1, sigmas(K),
        # alpha, mu0
        self.P = (K + 1) + 1 + (K - 1) + K + 1 + 1
        self.theta = anp.zeros((self.P, self.basis_dim))

    # ------------------------------ parameter utils ---------------------
    def _decode(self, eta: anp.ndarray) -> MixtureParams:
        phi = self.basis_fn(eta)  # (B,)
        raw = self.theta @ phi  # (P,)

        offset = 0
        logits = raw[offset : offset + self.K + 1]
        offset += self.K + 1
        m1 = raw[offset]
        offset += 1
        if self.K > 1:
            deltas = raw[offset : offset + self.K - 1]
            offset += self.K - 1
        else:
            deltas = anp.zeros(0)
        s_raw = raw[offset : offset + self.K]
        offset += self.K
        alpha_raw = raw[offset]
        offset += 1
        mu0_raw = raw[offset]

        # Link functions enforcing constraints
        weights = anp.exp(logits - anp.max(logits))
        weights = weights / anp.sum(weights)

        means = [m1]
        for d in deltas:
            means.append(means[-1] + anp.log1p(anp.exp(d)))  # softplus delta
        means = anp.array(means)

        sigmas = anp.log1p(anp.exp(s_raw)) + self.s_min
        alpha = 1.0 + anp.log1p(anp.exp(alpha_raw))
        mu0 = anp.log1p(anp.exp(mu0_raw)) + self.eps
        return MixtureParams(weights, means, sigmas, alpha, mu0)

    # ------------------------------ pdf utilities -----------------------
    def _lognormal_pdf(self, mu: anp.ndarray, m: float, s: float) -> anp.ndarray:
        return (
            1.0
            / (mu * s * anp.sqrt(2.0 * anp.pi))
            * anp.exp(-0.5 * ((anp.log(mu) - m) / s) ** 2)
        )

    def _pareto_pdf(self, mu: anp.ndarray, alpha: float, mu0: float) -> anp.ndarray:
        pdf = (alpha - 1.0) / mu0 * (mu0 / mu) ** alpha
        return anp.where(mu >= mu0, pdf, 0.0)

    def _mixture_pdf(self, mu: anp.ndarray, params: MixtureParams) -> anp.ndarray:
        comps: List[anp.ndarray] = []
        for k in range(self.K):
            comps.append(self._lognormal_pdf(mu, params.means[k], params.sigmas[k]))
        comps.append(self._pareto_pdf(mu, params.alpha, params.mu0))
        pdf = anp.zeros_like(mu)
        for w, p in zip(params.weights, comps):
            pdf = pdf + w * p
        # normalise numerically to protect against accumulation errors.
        # autograd does not provide a VJP for ``trapz`` so we expand the
        # trapezoidal rule manually.
        diffs = mu[1:] - mu[:-1]
        avg = 0.5 * (pdf[1:] + pdf[:-1])
        norm = anp.sum(avg * diffs)
        return pdf / (norm + 1e-12)

    # ------------------------------ loss -------------------------------
    def loss(self, theta_flat: anp.ndarray, data: List[Dict[str, anp.ndarray]]) -> float:
        theta = theta_flat.reshape(self.P, self.basis_dim)
        self.theta = theta
        total = 0.0
        for item in data:
            eta = item["eta"]
            y = item["y"]
            cnt = item["cnt"]
            N = item["N"]
            dmu = item["dmu"]

            params = self._decode(eta)
            mu = anp.exp(y)
            pdf = self._mixture_pdf(mu, params)
            lam = N * pdf * dmu
            lam = anp.clip(lam, 1e-30, anp.inf)
            total = total + anp.sum(lam - cnt * anp.log(lam))

            # Regularisers
            w = params.weights
            entropy = -anp.sum(w * anp.log(w + 1e-12))
            total = total + 1e-3 * entropy
            total = total + 0.05 * ((params.alpha - 3.0) ** 2 * anp.maximum(eta[2], 0))

        return total / len(data)

    # ------------------------------ training ---------------------------
    def fit(self, data: List[Dict[str, anp.ndarray]], maxiter: int = 50) -> None:
        theta0 = anp.zeros((self.P, self.basis_dim)).ravel()

        def obj(th):
            return self.loss(th, data)

        loss_grad = value_and_grad(obj)

        def func(th):
            l, g = loss_grad(th)
            return float(l), g

        res = minimize(func, theta0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
        self.theta = res.x.reshape(self.P, self.basis_dim)

    # ------------------------------ prediction ------------------------
    def pdf(self, mu: anp.ndarray, eta: anp.ndarray) -> anp.ndarray:
        params = self._decode(eta)
        return self._mixture_pdf(mu, params)


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def load_dataset(cache_dir: str, limit: int | None = None) -> List[Dict[str, anp.ndarray]]:
    """Load cached histograms produced by ``data_prep.py``."""

    import os
    import pandas as pd
    import numpy as np

    samples = pd.read_csv(os.path.join(cache_dir, "samples.csv"))
    if limit is not None:
        samples = samples.iloc[:limit]

    out: List[Dict[str, anp.ndarray]] = []
    for row in samples.itertuples(index=False):
        rid = row.rid
        path = os.path.join(cache_dir, "hists", f"{rid}.npz")
        if not os.path.exists(path):
            continue
        arr = np.load(path)
        if "logmu_mid" not in arr or "cnt_log" not in arr:
            continue
        y = arr["logmu_mid"].astype(float)
        cnt = arr["cnt_log"].astype(float)
        N = float(cnt.sum())
        dL = 0.01
        dmu = np.exp(y + 0.5 * dL) - np.exp(y - 0.5 * dL)
        eta = anp.array([row.kappa, row.gamma, row.s])
        out.append({"eta": eta, "y": y, "cnt": cnt, "N": N, "dmu": dmu, "rid": rid})
    return out


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_comparison(model: ConditionalMixtureModel, data: List[Dict[str, anp.ndarray]], out_path: str) -> None:
    """Plot empirical vs. model distributions for a list of samples."""

    import matplotlib.pyplot as plt
    import numpy as np

    n = len(data)
    cols = 5
    rows = int(anp.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=False)
    axes = axes.flat
    for ax, item in zip(axes, data):
        eta = item["eta"]
        y = item["y"]
        cnt = item["cnt"]
        N = item["N"]
        dmu = item["dmu"]
        rid = item["rid"]

        mu = np.exp(y)
        emp_pdf = cnt / N / dmu
        mdl_pdf = model.pdf(mu, eta)
        ax.plot(mu, emp_pdf, label="empirical")
        ax.plot(mu, mdl_pdf, label="model")
        ax.set_xscale("log")
        ax.set_title(rid)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
