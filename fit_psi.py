"""Fit cached histograms and save the resulting parameters.

This script loads histogram data prepared by ``data_prep.py`` and fits the
mixture model implemented in :mod:`mu_generator`.  The fitted parameters are
written to an ``.npz`` file that can later be consumed by
``load_psi_model.py`` to build an interpolator.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd

from mu_generator import load_interpolated, fit_single


def main(argv: List[str] | None = None) -> None:
    """Run the fitting pipeline and persist results."""

    parser = argparse.ArgumentParser(description="Fit histograms and save psi parameters")
    parser.add_argument(
        "--cache-dir",
        default="data_cache",
        help="directory containing histogram cache and samples.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optional limit on the number of histograms to fit",
    )
    parser.add_argument(
        "--out",
        default="psi_fits.npz",
        help="output filename for the fitted parameters",
    )
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

    psi_arr = np.vstack(psi_list)
    params_arr = np.vstack(params)
    np.savez(args.out, psi=psi_arr, params=params_arr)
    print(f"wrote {len(psi_arr)} fits to {args.out}")


if __name__ == "__main__":
    main()
