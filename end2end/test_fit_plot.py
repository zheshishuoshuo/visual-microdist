"""Train the end-to-end model and plot comparisons for 20 random samples."""

from __future__ import annotations

import argparse
import os

import pytest

pytest.skip("helper script, not a test", allow_module_level=True)

import autograd.numpy as anp
import numpy as np

from .model import ConditionalMixtureModel, load_dataset, plot_comparison


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit conditional model and plot comparisons")
    ap.add_argument("--cache-dir", default="data_cache", help="directory containing cached hists")
    ap.add_argument("--limit", type=int, default=20, help="number of samples to use")
    ap.add_argument("--out", default=os.path.join("end2end", "fit_vs_empirical.png"), help="output figure path")
    ap.add_argument("--maxiter", type=int, default=30, help="training iterations")
    args = ap.parse_args()

    data = load_dataset(args.cache_dir)
    if not data:
        raise RuntimeError("no data loaded; ensure cache directory is correct")
    rng = np.random.default_rng()
    n = min(args.limit, len(data))
    idx = rng.choice(len(data), size=n, replace=False)
    subset = [data[i] for i in idx]

    model = ConditionalMixtureModel(K=2)
    model.fit(subset, maxiter=args.maxiter)
    plot_comparison(model, subset, args.out)
    print(f"[OK] wrote comparison plot to {args.out}")


if __name__ == "__main__":
    main()
