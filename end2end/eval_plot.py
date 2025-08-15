"""Evaluate a trained conditional model and produce plots/metrics."""
from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np
import matplotlib.pyplot as plt

from .model import ConditionalMixtureModel, load_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate conditional model")
    ap.add_argument("--state", required=True, help="path to model_state.npz")
    ap.add_argument("--cache-dir", default="data_cache", help="dataset cache directory")
    ap.add_argument("--limit", type=int, default=20, help="number of samples to evaluate")
    ap.add_argument("--out", default=os.path.join("end2end", "eval.png"), help="output plot path")
    ap.add_argument("--csv", default=os.path.join("end2end", "metrics.csv"), help="output metrics csv")
    args = ap.parse_args()

    data = load_dataset(args.cache_dir, limit=args.limit)
    if not data:
        raise RuntimeError("no data loaded; ensure cache directory is correct")

    model = ConditionalMixtureModel.load_state(args.state)

    n = len(data)
    cols = 5
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=False)
    axes = axes.flat

    metrics = []
    for ax, item in zip(axes, data):
        eta = item["eta"]
        y = item["y"]
        cnt = item["cnt"]
        N = item["N"]
        dmu = item["dmu"]
        rid = item["rid"]

        mu = np.exp(y)
        pdf = model.pdf_mu(eta, mu)

        ax.plot(mu, cnt, label="cnt")
        ax.plot(mu, N * pdf * dmu, label="N*p*Δμ")
        ax.set_xscale("log")
        ax.set_title(rid)

        lam = np.clip(N * pdf * dmu, 1e-30, np.inf)
        nll = float(np.sum(lam - cnt * np.log(lam)) / (N + model.eps))
        leak = float(np.sum(pdf * dmu * (cnt == 0)))
        metrics.append({"rid": rid, "nll": nll, "leak": leak})

    for ax in axes[n:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rid", "nll", "leak"])
        writer.writeheader()
        writer.writerows(metrics)

    print(f"[OK] wrote plot to {args.out} and metrics to {args.csv}")


if __name__ == "__main__":
    main()
