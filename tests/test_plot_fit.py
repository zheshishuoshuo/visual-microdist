import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from end2end.stan_model import prepare_stan_data, fit_model, StanState, pdf_mu


def _make_dataset(root: str, n: int = 20) -> None:
    os.makedirs(os.path.join(root, 'hists'), exist_ok=True)
    rows = []
    logmu = np.linspace(-2, 3, 50)
    rng = np.random.default_rng(0)
    for i in range(n):
        eta = rng.uniform(size=3)
        rows.append({'rid': i, 'kappa': eta[0], 'gamma': eta[1], 's': eta[2]})
        cnt = rng.poisson(lam=5, size=logmu.size).astype(float)
        np.savez(os.path.join(root, 'hists', f'{i}.npz'), logmu_mid=logmu, cnt_log=cnt)
    pd.DataFrame(rows).to_csv(os.path.join(root, 'samples.csv'), index=False)


def test_stan_fit_plot(tmp_path):
    data_dir = tmp_path / 'cache'
    _make_dataset(str(data_dir))
    stan_data, aux = prepare_stan_data(str(data_dir), K=1)
    _, fit = fit_model(stan_data)
    state = StanState(fit=fit, aux=aux)
    samples = pd.read_csv(data_dir / 'samples.csv')
    mu = stan_data['mu']
    dmu = stan_data['dmu']
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    for ax, row, cnt, N in zip(axes.flat, samples.itertuples(), stan_data['cnt'], stan_data['Ntot']):
        eta = np.array([row.kappa, row.gamma, row.s])
        pdf = pdf_mu(eta, mu, state)
        ax.bar(mu, cnt, width=dmu, alpha=0.3, label='data')
        ax.plot(mu, N * pdf * dmu, label='fit')
        ax.set_xscale('log')
        ax.set_title(f"rid {row.rid}")
    for ax in axes.flat[samples.shape[0]:]:
        ax.axis('off')
    fig.tight_layout()
    out_file = tmp_path / 'stan_fit_vs_empirical.png'
    fig.savefig(out_file)
    assert out_file.exists()
