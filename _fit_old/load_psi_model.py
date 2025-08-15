"""Load saved psi fits and provide distribution utilities.

The ``fit_psi.py`` script saves mixture model parameters for many histogram
realizations.  This module reads those fits, constructs the
:class:`mu_generator.PsiModel` interpolator and offers a small command line
interface for evaluating or sampling ``p(mu | kappa, gamma, s)``.
"""

from __future__ import annotations

import argparse
from typing import Iterable, List

import numpy as np
import pandas as pd

from mu_generator import (
    PsiModel,
    analytic_cdf_mu,
    analytic_p_mu,
    build_psi_model,
    p_mu_given_eta,
    sample_mu,
)


def load_model(fits_file: str) -> PsiModel:
    """Load ``psi`` fits from ``fits_file`` and build a :class:`PsiModel`."""
    data = np.load(fits_file)
    psi = data["psi"]
    params = data["params"]
    df = pd.DataFrame(params, columns=["kappa", "gamma", "s"])
    model = build_psi_model(df, list(psi))
    return model


def main(argv: List[str] | None = None) -> None:
    """Small CLI for sampling or evaluating the PDF."""

    parser = argparse.ArgumentParser(description="Use saved psi fits")
    parser.add_argument("--fits", default="psi_fits.npz", help="file produced by fit_psi.py")
    parser.add_argument("--kappa", type=float, required=True, help="kappa value")
    parser.add_argument("--gamma", type=float, required=True, help="gamma value")
    parser.add_argument("--s", type=float, required=True, help="s value")
    parser.add_argument(
        "--mu", type=float, default=None, help="if given, evaluate PDF at this mu value"
    )
    parser.add_argument("--size", type=int, default=0, help="number of samples to draw")
    parser.add_argument("--seed", type=int, default=None, help="random seed for sampling")
    args = parser.parse_args(argv)

    model = load_model(args.fits)

    if args.mu is not None:
        val = p_mu_given_eta(np.array([args.mu]), args.kappa, args.gamma, args.s, model)
        print(val[0])
    elif args.size > 0:
        rng = np.random.default_rng(args.seed)
        samples = sample_mu(args.kappa, args.gamma, args.s, model, args.size, rng)
        for mu in samples:
            print(mu)
    else:
        parser.error("either --mu or --size must be specified")


__all__ = ["load_model", "analytic_p_mu", "analytic_cdf_mu", "p_mu_given_eta", "sample_mu", "PsiModel"]


if __name__ == "__main__":
    main()
