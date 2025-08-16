import os
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from end2end.stan_model import prepare_stan_data, pdf_mu, StanState


class FakeFit:
    """Minimal stand-in for :class:`CmdStanMLE` used in tests."""

    def __init__(self, P: int, K: int) -> None:
        self._vars = {
            "beta_m1": np.zeros(P),
            "beta_m_diff": np.zeros((P, K - 1)),
            "beta_logsig": np.zeros((P, K)),
            "beta_w": np.zeros((P, K + 1)),
            "beta_alpha": np.zeros(P),
            "beta_mu0": np.zeros(P),
        }

    def stan_variable(self, name: str):
        return self._vars[name]


def _make_dataset(root: str) -> np.ndarray:
    os.makedirs(os.path.join(root, "hists"), exist_ok=True)
    # single sample
    df = pd.DataFrame([
        {"rid": 0, "kappa": 0.5, "gamma": 0.1, "s": 0.8},
    ])
    df.to_csv(os.path.join(root, "samples.csv"), index=False)

    logmu = np.linspace(-2, 9, 20)
    cnt = np.random.poisson(lam=5, size=logmu.shape[0]).astype(float)
    np.savez(os.path.join(root, "hists", "0.npz"), logmu_mid=logmu, cnt_log=cnt)
    return df.loc[0, ["kappa", "gamma", "s"]].to_numpy()


def test_pdf_mu_basic() -> None:
    with TemporaryDirectory() as d:
        eta = _make_dataset(d)
        stan_data, aux = prepare_stan_data(d)
        fake_fit = FakeFit(stan_data["P"], stan_data["K"])
        state = StanState(fit=fake_fit, aux=aux)

        mu = stan_data["mu"]
        pdf = pdf_mu(eta, mu, state)

        assert pdf.shape == mu.shape
        assert np.all(pdf >= 0)
        integral = np.trapz(pdf, mu)
        assert np.isclose(integral, 1.0, atol=1e-1)
