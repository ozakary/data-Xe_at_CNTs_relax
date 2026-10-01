#!/usr/bin/env python3
"""
Step 2b - TCF diagnostics: multi-timescale view plus the running-integral
convergence check that actually tells you how far the trajectory needs to
resolve the decay before J(0) (and hence T1, T2 in the extreme-narrowing
limit) can be trusted.

Nothing here recomputes the correlation function, it's already in
Step 2's output .npz out to the full trajectory length. This just looks at
more of it, and computes:

  - a short-lag view (same as Step 2's default plot)
  - a long-lag linear view, with the noise floor drawn as a reference line
  - a log-time view spanning the whole trajectory, for seeing decay on
    multiple timescales at once
  - the running integral of TCF(t) up to lag T, as a function of T. For a
    simple exponential decay this equals TCF(0)*tau_c once T >> tau_c, so
    where this curve flattens is a direct, model-free readout of how far
    the decay needs to be trusted, and an estimate of J(0) up to a factor
    of TCF(0).

Usage
-----
    python step2b_tcf_diagnostics.py \\
        --tcf ./cnt_10_0_u23_Xe4_10ns_tcf.npz \\
        --out-prefix ./cnt_10_0_u23_Xe4_10ns_tcf_diag \\
        --long-lag-ps 2000 \\
        --noise-floor-frac 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tcf", required=True, type=Path, help="Step 2 output .npz")
    ap.add_argument("--out-prefix", required=True, type=Path)
    ap.add_argument("--short-lag-ps", type=float, default=20.0)
    ap.add_argument("--long-lag-ps", type=float, default=2000.0)
    ap.add_argument("--noise-floor-frac", type=float, default=0.1, help="fraction of the longest lags used to estimate the noise floor")
    args = ap.parse_args()

    d = np.load(args.tcf)
    lag_fs = d["lag_fs"]
    tcf = d["tcf_total"]
    lag_ps = lag_fs / 1000.0
    n = len(tcf)

    n_tail = max(1, int(n * args.noise_floor_frac))
    noise_floor = tcf[-n_tail:].mean()
    noise_std = tcf[-n_tail:].std()
    print(f"Noise floor (last {args.noise_floor_frac*100:.0f}% of lags): {noise_floor:.4e} +/- {noise_std:.4e}")
    print(f"TCF(0) = {tcf[0]:.4e}, ratio TCF(0)/noise_floor = {tcf[0]/noise_floor:.1f}")

    for probe_ps in [args.short_lag_ps, 100.0, 500.0, args.long_lag_ps, lag_ps[-1]]:
        idx = np.searchsorted(lag_ps, probe_ps)
        idx = min(idx, n - 1)
        ratio = tcf[idx] / noise_floor
        print(f"  TCF at {lag_ps[idx]:8.2f} ps = {tcf[idx]:.4e}  ({ratio:.2f} x noise floor)")

    # running integral (ps units on the time axis, values carry through to J(0) up to that unit choice)
    running_integral = np.concatenate([[0.0], np.cumsum((tcf[1:] + tcf[:-1]) / 2.0 * np.diff(lag_ps))])
    print()
    print("Running integral of TCF (proportional to J(0) in the plateau region), by upper limit:")
    for probe_ps in [args.short_lag_ps, 100.0, 500.0, args.long_lag_ps, lag_ps[-1]]:
        idx = np.searchsorted(lag_ps, probe_ps)
        idx = min(idx, n - 1)
        print(f"  integral to {lag_ps[idx]:8.2f} ps = {running_integral[idx]:.4e}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    m = lag_ps <= args.short_lag_ps
    ax.plot(lag_ps[m], tcf[m], lw=1)
    ax.axhline(noise_floor, color="gray", ls="--", lw=1, label="noise floor")
    ax.set_xlabel("time (ps)"); ax.set_ylabel("TCF"); ax.set_title(f"Short lag (0-{args.short_lag_ps:.0f} ps)")
    ax.legend()

    ax = axes[0, 1]
    m = lag_ps <= args.long_lag_ps
    ax.plot(lag_ps[m], tcf[m], lw=1)
    ax.axhline(noise_floor, color="gray", ls="--", lw=1, label="noise floor")
    ax.set_xlabel("time (ps)"); ax.set_ylabel("TCF"); ax.set_title(f"Long lag (0-{args.long_lag_ps:.0f} ps)")
    ax.legend()

    ax = axes[1, 0]
    m = lag_ps > 0
    ax.plot(lag_ps[m], tcf[m], lw=1)
    ax.axhline(noise_floor, color="gray", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("time (ps, log scale)"); ax.set_ylabel("TCF"); ax.set_title("Full trajectory, log-time")

    ax = axes[1, 1]
    ax.plot(lag_ps[1:], running_integral[1:], lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("upper limit (ps, log scale)"); ax.set_ylabel(r"$\int_0^T$ TCF dt")
    ax.set_title("Running integral, convergence = trustworthy J(0)")

    fig.tight_layout()
    out_path = Path(f"{args.out_prefix}.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
