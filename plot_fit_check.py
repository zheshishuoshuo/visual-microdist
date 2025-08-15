"""Visual check of fitted curves against histogram data."""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

import mu_generator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot histogram data with fitted analytic curves"
    )
    parser.add_argument(
        "--cache-dir",
        default="data_cache",
        help="Directory containing cached histograms",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=4,
        help="Number of random histograms to display",
    )
    parser.add_argument(
        "--outfile",
        help="If given, save the figure to this path"
    )
    args = parser.parse_args()

    grid, items = mu_generator.load_interpolated(args.cache_dir)
    rng = np.random.default_rng()
    n = min(args.num, len(items))
    idx = rng.choice(len(items), size=n, replace=False)
    picks = [items[i] for i in idx]

    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    dL = np.median(np.diff(grid))
    mu = np.exp(grid)
    dmu = np.exp(grid + 0.5 * dL) - np.exp(grid - 0.5 * dL)

    for ax, item in zip(axes, picks):
        cnt = item["cnt"]
        N = item["N"]
        psi = mu_generator.fit_single(grid, cnt, N)
        model = N * mu_generator.analytic_p_mu(mu, psi) * dmu
        ax.plot(mu, cnt, drawstyle="steps-mid", label="data")
        ax.plot(mu, model, label="fit")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(item["rid"])
        ax.legend(fontsize="small")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()

    if args.outfile:
        fig.savefig(args.outfile)
    else:
        plt.show()


if __name__ == "__main__":
    main()
