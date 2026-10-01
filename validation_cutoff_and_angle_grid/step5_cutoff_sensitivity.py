#!/usr/bin/env python3
"""
Step 5 - Cutoff sensitivity check.

Steps 3/4 fit the TCF out to a single, by-eye chosen fit-max-lag-ps. This
reruns that fit (on the pooled, 4-track-averaged TCF, same as Step 3) across
a range of cutoffs and reports T1, T2 at each, so the choice of cutoff is a
demonstrated, stable result rather than an unexamined pick. A genuine
plateau in T1(cutoff)/T2(cutoff) as the cutoff grows past the slow
component's correlation time is what you want to see and cite; drift that
never settles means the fit region needs to be reconsidered before trusting
Step 3/4's numbers.

Usage
-----
    python step5_cutoff_sensitivity.py \\
        --tcf ./cnt_10_0_u23_Xe4_10ns_tcf.npz \\
        --cutoffs-ps 300 500 750 1000 1500 2000 3000 \\
        --b0-fields 9.4 14.1 \\
        --out ./cnt_10_0_u23_Xe4_10ns_cutoff_sensitivity.npz \\
        --plot ./cnt_10_0_u23_Xe4_10ns_cutoff_sensitivity.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


def mono_exp(t, A, tau):
    return A * np.exp(-t / tau)


def bi_exp(t, A1, tau1, A2, tau2):
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)


def lorentzian_sum(omega, amps, taus):
    omega = np.atleast_1d(omega).astype(float)
    J = np.zeros_like(omega)
    for A, tau in zip(amps, taus):
        J += A * tau / (1.0 + (omega * tau) ** 2)
    return J


def fit_and_relax(t_ps, y, gamma_xe, b0_fields):
    p0_mono = [y[0], 2.0]
    popt_mono, _ = curve_fit(mono_exp, t_ps, y, p0=p0_mono, maxfev=20000)
    try:
        p0_bi = [y[0] * 0.8, 1.0, y[0] * 0.2, 50.0]
        popt_bi, _ = curve_fit(bi_exp, t_ps, y, p0=p0_bi, maxfev=50000,
                                bounds=([-np.inf, 1e-3, -np.inf, 1e-3], [np.inf, np.inf, np.inf, np.inf]))
        resid = y - bi_exp(t_ps, *popt_bi)
        r2 = 1.0 - np.sum(resid**2) / np.sum((y - y.mean())**2)
        amps_ps, taus_ps = [popt_bi[0], popt_bi[2]], [popt_bi[1], popt_bi[3]]
    except RuntimeError:
        resid = y - mono_exp(t_ps, *popt_mono)
        r2 = 1.0 - np.sum(resid**2) / np.sum((y - y.mean())**2)
        amps_ps, taus_ps = [popt_mono[0]], [popt_mono[1]]

    taus_si = [tau * 1e-12 for tau in taus_ps]
    J0 = lorentzian_sum(0.0, amps_ps, taus_si)[0]
    out = {"r2": r2, "taus_ps": taus_ps, "J0": J0}
    for B0 in b0_fields:
        w0 = abs(gamma_xe) * B0
        Jw0 = lorentzian_sum(w0, amps_ps, taus_si)[0]
        out[f"T1_{B0}"] = 1.0 / (6.0 * B0**2 * Jw0)
        out[f"T2_{B0}"] = 1.0 / (B0**2 * (4.0 * J0 + 3.0 * Jw0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tcf", required=True, type=Path)
    ap.add_argument("--cutoffs-ps", type=float, nargs="+", default=[300, 500, 750, 1000, 1500, 2000, 3000])
    ap.add_argument("--b0-fields", type=float, nargs="+", default=[9.4, 14.1])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.tcf)
    lag_fs = d["lag_fs"]
    tcf = d["tcf_total"]
    gamma_xe = float(d["gamma_xe"])
    lag_ps = lag_fs / 1000.0

    print(f"{'cutoff(ps)':>10} {'R^2':>6} {'tau1(ps)':>9} {'tau2(ps)':>9}", end="")
    for B0 in args.b0_fields:
        print(f" {'T1@'+str(B0)+'T':>12} {'T2@'+str(B0)+'T':>12}", end="")
    print()

    rows = []
    for cutoff in args.cutoffs_ps:
        mask = lag_ps <= cutoff
        if mask.sum() < 20:
            print(f"{cutoff:10.0f}  (skipped, not enough points)")
            continue
        res = fit_and_relax(lag_ps[mask], tcf[mask], gamma_xe, args.b0_fields)
        taus = res["taus_ps"] + [np.nan] * (2 - len(res["taus_ps"]))
        print(f"{cutoff:10.0f} {res['r2']:6.3f} {taus[0]:9.3f} {taus[1]:9.3f}", end="")
        for B0 in args.b0_fields:
            print(f" {res[f'T1_{B0}']:12.4e} {res[f'T2_{B0}']:12.4e}", end="")
        print()
        rows.append((cutoff, res))

    np.savez_compressed(args.out, cutoffs_ps=args.cutoffs_ps, b0_fields=args.b0_fields,
                         rows=[r for _, r in rows])
    print(f"\nWrote {args.out}")

    if args.plot and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cutoffs = [c for c, _ in rows]
        fig, axes = plt.subplots(1, len(args.b0_fields), figsize=(5 * len(args.b0_fields), 4), squeeze=False)
        for i, B0 in enumerate(args.b0_fields):
            ax = axes[0, i]
            T1s = [r[f"T1_{B0}"] for _, r in rows]
            T2s = [r[f"T2_{B0}"] for _, r in rows]
            ax.plot(cutoffs, T1s, "o-", label="T1")
            ax.plot(cutoffs, T2s, "s-", label="T2")
            ax.set_xlabel("fit cutoff (ps)")
            ax.set_ylabel("time (s)")
            ax.set_title(f"B0 = {B0} T")
            ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
