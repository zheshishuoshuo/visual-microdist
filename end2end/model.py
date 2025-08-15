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
from typing import Callable, Dict, List, Tuple

import autograd.numpy as anp
from autograd import value_and_grad
from scipy.optimize import minimize
import numpy as np


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
    y_lo: float
    y_hi: float


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
        # feature normalisation stats (set during ``fit``)
        self.feat_mean: anp.ndarray | None = None
        self.feat_std: anp.ndarray | None = None
        self.y_grid: anp.ndarray | None = None
        self.dL: float | None = None

        self.basis_dim = len(basis_fn(anp.array([0.0, 0.0, 0.0])))
        # number of parameter groups: logits(K+1), m1, Δm for k>1, sigmas(K),
        # alpha, mu0, y_lo, Δy
        self.P = (K + 1) + 1 + (K - 1) + K + 1 + 1 + 2
        self.theta = anp.zeros((self.P, self.basis_dim))

    # ------------------------------ parameter utils ---------------------
    def _decode(self, eta: anp.ndarray) -> MixtureParams:
        phi = self.basis_fn(eta)  # (B,)
        if self.feat_mean is not None and self.feat_std is not None:
            phi = (phi - self.feat_mean) / (self.feat_std + 1e-12)
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
        offset += 1
        ylo_raw = raw[offset]
        offset += 1
        dy_raw = raw[offset]

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
        y_lo = ylo_raw
        y_hi = y_lo + anp.log1p(anp.exp(dy_raw))  # ensure y_hi > y_lo
        return MixtureParams(weights, means, sigmas, alpha, mu0, y_lo, y_hi)

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

    def _mixture_pdf(self, y: anp.ndarray, params: MixtureParams) -> anp.ndarray:
        """Mixture density on a ``log μ`` grid with soft support gating."""

        mu = anp.exp(y)
        comps: List[anp.ndarray] = []
        for k in range(self.K):
            comps.append(self._lognormal_pdf(mu, params.means[k], params.sigmas[k]))
        comps.append(self._pareto_pdf(mu, params.alpha, params.mu0))
        pdf = anp.zeros_like(mu)
        for w, p in zip(params.weights, comps):
            pdf = pdf + w * p

        # Soft support window during training
        beta = 100.0  # sharpness of the sigmoid gates
        gate_lo = 1.0 / (1.0 + anp.exp(-beta * (y - params.y_lo)))
        gate_hi = 1.0 / (1.0 + anp.exp(-beta * (params.y_hi - y)))
        gate = gate_lo * gate_hi
        pdf = pdf * gate

        # normalise numerically to protect against accumulation errors.
        diffs = mu[1:] - mu[:-1]
        avg = 0.5 * (pdf[1:] + pdf[:-1])
        norm = anp.sum(avg * diffs)
        return pdf / (norm + 1e-12)

    # ------------------------------ loss -------------------------------
    def loss(self, theta_flat: anp.ndarray, data: List[Dict[str, anp.ndarray]]) -> float:
        """Per-sample averaged Poisson NLL with additional penalties."""

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
            pdf = self._mixture_pdf(y, params)
            lam = N * pdf * dmu
            lam = anp.clip(lam, 1e-30, anp.inf)
            nll = anp.sum(lam - cnt * anp.log(lam)) / (N + self.eps)

            # Regularisers
            w = params.weights
            entropy = -anp.sum(w * anp.log(w + 1e-12))
            ent_term = -1e-3 * entropy
            prior = 0.02 * ((params.alpha - 3.0) ** 2 * anp.maximum(eta[2], 0))
            leak = 100.0 * anp.sum(pdf * dmu * (cnt == 0))

            total = total + nll + ent_term + prior + leak

        return total / len(data)

    # ------------------------------ training ---------------------------
    def fit(self, data: List[Dict[str, anp.ndarray]], maxiter: int = 50) -> None:
        # compute feature normalisation statistics
        feats = anp.array([self.basis_fn(item["eta"]) for item in data])
        self.feat_mean = anp.mean(feats, axis=0)
        self.feat_std = anp.std(feats, axis=0) + 1e-6

        # assume shared y grid
        self.y_grid = data[0]["y"]
        if len(self.y_grid) > 1:
            self.dL = float(self.y_grid[1] - self.y_grid[0])
        else:
            self.dL = 0.01

        theta0 = anp.zeros((self.P, self.basis_dim)).ravel()

        def obj(th):
            return self.loss(th, data)

        loss_grad = value_and_grad(obj)

        def func(th):
            l, g = loss_grad(th)
            return float(l), g

        res = minimize(
            func,
            theta0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter},
        )
        self.theta = res.x.reshape(self.P, self.basis_dim)

        # diagnostics
        for i, item in enumerate(data):
            eta = item["eta"]
            y = item["y"]
            cnt = item["cnt"]
            N = item["N"]
            dmu = item["dmu"]
            params = self._decode(eta)
            pdf = self._mixture_pdf(y, params)
            lam = N * pdf * dmu
            lam = anp.clip(lam, 1e-30, anp.inf)
            nll = float(anp.sum(lam - cnt * anp.log(lam)) / (N + self.eps))
            w = params.weights
            entropy = float(1e-3 * (-anp.sum(w * anp.log(w + 1e-12))))
            prior = float(0.02 * ((params.alpha - 3.0) ** 2 * anp.maximum(eta[2], 0)))
            leak = float(100.0 * anp.sum(pdf * dmu * (cnt == 0)))
            print(
                f"sample {i}: nll={nll:.3f}, leak={leak:.3f}, entropy={entropy:.3f}, prior={prior:.3f}"
            )

    # ------------------------------ prediction ------------------------
    def pdf_mu(self, eta: anp.ndarray, mu: anp.ndarray) -> anp.ndarray:
        """Return ``p(μ | η)`` on ``mu`` grid with hard support applied."""

        y = anp.log(mu)
        params = self._decode(eta)
        pdf = self._mixture_pdf(y, params)
        mask = (y >= params.y_lo) * (y <= params.y_hi)
        pdf = pdf * mask
        if anp.sum(pdf) == 0:
            return pdf
        # Renormalise after hard masking
        diffs = mu[1:] - mu[:-1]
        avg = 0.5 * (pdf[1:] + pdf[:-1])
        norm = anp.sum(avg * diffs)
        return pdf / (norm + 1e-12)

    # backward compatible alias
    def pdf(self, mu: anp.ndarray, eta: anp.ndarray) -> anp.ndarray:  # pragma: no cover - deprecated
        return self.pdf_mu(eta, mu)

    def sample_mu(
        self, eta: anp.ndarray, size: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Draw samples from ``p(μ|η)``."""

        rng = np.random.default_rng() if rng is None else rng
        params = self._decode(eta)
        comps = rng.choice(self.K + 1, size=size, p=np.asarray(params.weights))
        out = np.empty(size)
        for k in range(self.K):
            mask = comps == k
            if np.any(mask):
                out[mask] = rng.lognormal(
                    mean=float(params.means[k]),
                    sigma=float(params.sigmas[k]),
                    size=mask.sum(),
                )
        mask = comps == self.K
        if np.any(mask):
            u = rng.random(mask.sum())
            out[mask] = params.mu0 * (1 - u) ** (-1.0 / (params.alpha - 1.0))
        return out

    def save_state(self, path: str) -> None:
        """Serialise model parameters to ``path``."""

        np.savez(
            path,
            theta=np.asarray(self.theta),
            feat_mean=np.asarray(self.feat_mean),
            feat_std=np.asarray(self.feat_std),
            y_grid=np.asarray(self.y_grid) if self.y_grid is not None else None,
            dL=self.dL if self.dL is not None else 0.0,
            config=np.array([self.K, self.s_min, self.eps]),
        )

    @classmethod
    def load_state(cls, path: str) -> "ConditionalMixtureModel":
        """Load model parameters from ``path``."""

        arr = np.load(path, allow_pickle=True)
        K, s_min, eps = arr["config"]
        model = cls(int(K), s_min=float(s_min), eps=float(eps))
        model.theta = arr["theta"]
        model.feat_mean = arr["feat_mean"]
        model.feat_std = arr["feat_std"]
        model.y_grid = arr["y_grid"] if "y_grid" in arr else None
        model.dL = float(arr["dL"]) if "dL" in arr else None
        return model


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


def fit(
    cache_dir: str,
    limit: int | None = None,
    K: int = 2,
    maxiter: int = 50,
    out_path: str = "model_state.npz",
) -> str:
    """Fit the model on cached histograms and persist its state."""

    data = load_dataset(cache_dir, limit=limit)
    if not data:
        raise RuntimeError("no data loaded; ensure cache directory is correct")
    model = ConditionalMixtureModel(K=K)
    model.fit(data, maxiter=maxiter)
    model.save_state(out_path)
    return out_path


def pdf_mu(eta: anp.ndarray, mu_grid: anp.ndarray, state_path: str) -> anp.ndarray:
    """Compute ``p(μ|η)`` on ``mu_grid`` using a saved model state."""

    model = ConditionalMixtureModel.load_state(state_path)
    return model.pdf_mu(eta, mu_grid)


def sample_mu(eta: anp.ndarray, size: int, state_path: str) -> np.ndarray:
    """Draw ``size`` samples from ``p(μ|η)`` using a saved model state."""

    model = ConditionalMixtureModel.load_state(state_path)
    return model.sample_mu(eta, size)


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
        mdl_pdf = model.pdf_mu(eta, mu)
        ax.plot(mu, cnt, label="cnt")
        ax.plot(mu, N * mdl_pdf * dmu, label="N*p*Δμ")
        ax2 = ax.twinx()
        ax2.plot(mu, cnt / N, color="C2", label="q")
        ax2.plot(mu, mdl_pdf * dmu, color="C3", label="p*Δμ")
        ax.set_xscale("log")
        ax.set_title(rid)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = [], []
    for ax in axes[: min(n, len(axes))]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
