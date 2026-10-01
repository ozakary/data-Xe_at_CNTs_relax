#!/usr/bin/env python3
"""
Step 6 (new) - Cutoff robustness for the powder-averaged methodology,
across the full 38-system sweep.

Same role as step6_check_cutoff_robustness.py, but for the powder-averaged
T1, T2 rather than the theta=0-only fit, reusing
step2g_cutoff_robustness_powder.py's per-system logic (build each angle's
TCF once, refit at every cutoff) across every system in the manifest.

Confirmed on two structurally different reference systems before running
this across all 38, (10,0) (the harder case, converged by ~750 ps) and
(30,0) (converged by ~50 ps), both well within fit-max-lag-ps=3000, see the
conversation record. This script exists to confirm that holds everywhere,
not just the two systems already checked by hand.

Output
------
  --out-detailed : one row per (system, B0, cutoff): T1, T2, and percent
                    difference from that system's own 3000 ps powder average
  --out-summary  : one row per (system, B0): the shortest cutoff at which
                    T1 (and everything longer) stays within --tolerance-pct
                    of the 3000 ps reference, or NaN if nothing tested
                    achieves that

Usage
-----
    python step6_check_cutoff_robustness_powder.py \\
        --manifest ./run_manifest.csv \\
        --results-dir ./results \\
        --n-angles 10 \\
        --cutoffs-ps 20 50 100 200 300 500 750 1000 1500 2000 3000 \\
        --tolerance-pct 10 \\
        --b0-fields 9.4 14.1 \\
        --out-detailed ./powder_cutoff_robustness_detailed.csv \\
        --out-summary ./powder_cutoff_robustness_summary.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm

GAMMA_XE_DEFAULT = -7.451956e7  # rad s^-1 T^-1, 129Xe gyromagnetic ratio, matches step2_build_tcf.py
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Verbatim from step2g_cutoff_robustness_powder.py, copied rather than
# imported so this script has no dependency on that file.
# ---------------------------------------------------------------------------

def compute_h0_h1(tensor, gamma_xe):
    xz = tensor[..., 0, 2]
    yz = tensor[..., 1, 2]
    trace = tensor[..., 0, 0] + tensor[..., 1, 1] + tensor[..., 2, 2]
    sigma_iso = trace / 3.0
    zz_traceless_ppm = tensor[..., 2, 2] - sigma_iso
    h0 = 0.5 * gamma_xe * (zz_traceless_ppm * 1e-6)
    h1 = (1.0 / np.sqrt(6.0)) * gamma_xe * (-(xz * 1e-6) - 1j * (yz * 1e-6))
    return h0, h1


def compute_h2(tensor, gamma_xe):
    xx = tensor[..., 0, 0]
    yy = tensor[..., 1, 1]
    xy = tensor[..., 0, 1]
    return (gamma_xe / np.sqrt(6.0)) * (0.5 * (xx - yy) * 1e-6 + 1j * (xy * 1e-6))


def fft_corr(x):
    n = x.shape[0]
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    X = np.fft.fft(x, n=nfft)
    S = np.conj(X) * X
    c = np.fft.ifft(S)[:n]
    counts = np.arange(n, 0, -1)
    return c / counts


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


def fit_quality(t, y, model, popt):
    resid = y - model(t, *popt)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def wigner_d2_row0(theta):
    s, c2 = math.sin(theta), math.cos(2 * theta)
    s2 = math.sin(2 * theta)
    return np.array([0.25 * math.sqrt(6) * s ** 2, -0.25 * math.sqrt(6) * s2,
                      0.75 * c2 + 0.25, 0.25 * math.sqrt(6) * s2, 0.25 * math.sqrt(6) * s ** 2])


def wigner_d2_row1(theta):
    s, c = math.sin(theta), math.cos(theta)
    c2 = math.cos(2 * theta)
    return np.array([0.5 * (c - 1.0) * s, 0.5 * c - 0.5 * c2, -0.25 * math.sqrt(6) * math.sin(2 * theta),
                      0.5 * c + 0.5 * c2, 0.5 * (c + 1.0) * s])


def tube_frame_components(tensor, gamma_xe):
    h0, h1 = compute_h0_h1(tensor, gamma_xe)
    h2 = compute_h2(tensor, gamma_xe)
    return [np.conj(h2), -np.conj(h1), h0.astype(complex), h1, h2]  # k = -2,-1,0,1,2


def lab_frame_h0_h1(tensor, gamma_xe, theta_rad):
    tube = tube_frame_components(tensor, gamma_xe)
    row0, row1 = wigner_d2_row0(theta_rad), wigner_d2_row1(theta_rad)
    h_lab_0 = sum(c * h for c, h in zip(row0, tube))
    h_lab_1 = sum(c * h for c, h in zip(row1, tube))
    return h_lab_0, h_lab_1


def fit_pooled(t_ps, y):
    amp_bound = 5.0 * max(np.max(np.abs(y)), 1e-10)
    p0_mono = [abs(y[0]), 2.0]
    popt_mono, _ = curve_fit(mono_exp, t_ps, y, p0=p0_mono, maxfev=20000,
                              bounds=([0.0, 1e-3], [amp_bound, np.inf]))
    r2_mono = fit_quality(t_ps, y, mono_exp, popt_mono)
    try:
        p0_bi = [abs(y[0]) * 0.8, 1.0, abs(y[0]) * 0.2, 50.0]
        popt_bi, _ = curve_fit(bi_exp, t_ps, y, p0=p0_bi, maxfev=50000,
                                bounds=([0.0, 1e-3, 0.0, 1e-3], [amp_bound, np.inf, amp_bound, np.inf]))
        r2_bi = fit_quality(t_ps, y, bi_exp, popt_bi)
        return [popt_bi[0], popt_bi[2]], [popt_bi[1], popt_bi[3]], r2_bi
    except RuntimeError:
        return [popt_mono[0]], [popt_mono[1]], r2_mono


def powder_average_rate(theta_deg_grid, quantity_values):
    theta_rad = np.radians(np.asarray(theta_deg_grid, dtype=float))
    rates = 1.0 / np.asarray(quantity_values, dtype=float)
    weighted_rate = _trapz(rates * np.sin(theta_rad), theta_rad)
    return 1.0 / weighted_rate


def relax_from_fit(amps_ps, taus_ps, gamma_xe, b0_fields):
    taus_si = [tau * 1e-12 for tau in taus_ps]
    J0 = lorentzian_sum(0.0, amps_ps, taus_si)[0]
    out = {}
    for B0 in b0_fields:
        w0 = abs(gamma_xe) * B0
        Jw0 = lorentzian_sum(w0, amps_ps, taus_si)[0]
        T1 = 1.0 / (6.0 * B0 ** 2 * Jw0) if Jw0 > 0 else float("nan")
        T2 = 1.0 / (B0 ** 2 * (4.0 * J0 + 3.0 * Jw0)) if (4.0 * J0 + 3.0 * Jw0) > 0 else float("nan")
        out[B0] = dict(T1=T1, T2=T2)
    return out


def run_one_system(assembled_path, gamma_xe, n_angles, cutoffs_ps, b0_fields):
    """Returns {cutoff: {B0: {T1, T2}}} for one system, building each
    angle's TCF once and refitting at every cutoff, exactly as
    step2g_cutoff_robustness_powder.py does."""
    d = np.load(assembled_path)
    tensor = d["tensor"]
    dt_fs = float(d["dt_fs"])
    theta_grid_deg = list(np.linspace(0.0, 90.0, n_angles))
    lag_ps_full = np.arange(tensor.shape[0]) * dt_fs / 1000.0

    tcf_by_theta = {}
    for theta_deg in theta_grid_deg:
        h_lab_0, h_lab_1 = lab_frame_h0_h1(tensor, gamma_xe, math.radians(theta_deg))
        dh0 = h_lab_0 - h_lab_0.mean(axis=0)[None, :]
        dh1 = h_lab_1 - h_lab_1.mean(axis=0)[None, :]
        n_frames, n_atoms = tensor.shape[0], tensor.shape[1]
        tcf_h0 = np.empty((n_frames, n_atoms))
        tcf_h1 = np.empty((n_frames, n_atoms), dtype=complex)
        for k in range(n_atoms):
            tcf_h0[:, k] = fft_corr(dh0[:, k]).real
            tcf_h1[:, k] = fft_corr(dh1[:, k])
        tcf_by_theta[theta_deg] = tcf_h0.mean(axis=1) + 2.0 * tcf_h1.mean(axis=1).real

    powder_by_cutoff = {}
    for cutoff in sorted(cutoffs_ps):
        mask = lag_ps_full <= cutoff
        if mask.sum() < 20:
            continue
        per_theta_results = {}
        for theta_deg in theta_grid_deg:
            y_fit = tcf_by_theta[theta_deg][mask]
            amps_ps, taus_ps, _ = fit_pooled(lag_ps_full[mask], y_fit)
            per_theta_results[theta_deg] = relax_from_fit(amps_ps, taus_ps, gamma_xe, b0_fields)
        powder = {}
        for B0 in b0_fields:
            T1s = [per_theta_results[t][B0]["T1"] for t in theta_grid_deg]
            T2s = [per_theta_results[t][B0]["T2"] for t in theta_grid_deg]
            powder[B0] = dict(T1=powder_average_rate(theta_grid_deg, T1s),
                               T2=powder_average_rate(theta_grid_deg, T2s))
        powder_by_cutoff[cutoff] = powder
    return powder_by_cutoff


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--gamma-xe", type=float, default=GAMMA_XE_DEFAULT)
    ap.add_argument("--n-angles", type=int, default=10)
    ap.add_argument("--cutoffs-ps", type=float, nargs="+",
                     default=[20, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000])
    ap.add_argument("--reference-cutoff-ps", type=float, default=3000.0)
    ap.add_argument("--tolerance-pct", type=float, default=10.0)
    ap.add_argument("--b0-fields", type=float, nargs="+", default=[9.4, 14.1])
    ap.add_argument("--out-detailed", required=True, type=Path)
    ap.add_argument("--out-summary", required=True, type=Path)
    args = ap.parse_args()

    if args.reference_cutoff_ps not in args.cutoffs_ps:
        args.cutoffs_ps = sorted(set(args.cutoffs_ps) | {args.reference_cutoff_ps})

    manifest = pd.read_csv(args.manifest)
    detailed_rows, summary_rows = [], []
    n_missing = 0

    for _, row in tqdm(list(manifest.iterrows()), desc="Systems"):
        run_id = row["run_id"]
        assembled = args.results_dir / run_id / f"{run_id}_assembled.npz"
        if not assembled.exists():
            n_missing += 1
            continue

        meta = row.to_dict()
        powder_by_cutoff = run_one_system(assembled, args.gamma_xe, args.n_angles,
                                           args.cutoffs_ps, args.b0_fields)
        if args.reference_cutoff_ps not in powder_by_cutoff:
            n_missing += 1
            continue
        ref = powder_by_cutoff[args.reference_cutoff_ps]

        per_b0_diffs = {B0: [] for B0 in args.b0_fields}
        for cutoff in sorted(powder_by_cutoff):
            pw = powder_by_cutoff[cutoff]
            for B0 in args.b0_fields:
                T1_ref, T2_ref = ref[B0]["T1"], ref[B0]["T2"]
                T1c, T2c = pw[B0]["T1"], pw[B0]["T2"]
                T1_pct = 100.0 * (T1c - T1_ref) / T1_ref if np.isfinite(T1c) and T1_ref else np.nan
                T2_pct = 100.0 * (T2c - T2_ref) / T2_ref if np.isfinite(T2c) and T2_ref else np.nan
                detailed_rows.append({
                    **meta, "B0_T": B0, "cutoff_ps": cutoff,
                    "T1_s": T1c, "T2_s": T2c, "T1_ref_s": T1_ref, "T2_ref_s": T2_ref,
                    "T1_pct_diff": T1_pct, "T2_pct_diff": T2_pct,
                })
                per_b0_diffs[B0].append((cutoff, T1_pct))

        for B0 in args.b0_fields:
            diffs = per_b0_diffs[B0]
            min_stable_cutoff = np.nan
            for i, (cutoff, _) in enumerate(diffs):
                if all(abs(d_) <= args.tolerance_pct for _, d_ in diffs[i:] if np.isfinite(d_)):
                    min_stable_cutoff = cutoff
                    break
            summary_rows.append({
                **meta, "B0_T": B0,
                "min_stable_cutoff_ps": min_stable_cutoff,
                "shortest_cutoff_tested_ps": diffs[0][0],
                "pct_diff_at_shortest_cutoff": diffs[0][1],
                "stable_at_shortest_cutoff": bool(np.isfinite(diffs[0][1]) and abs(diffs[0][1]) <= args.tolerance_pct),
            })

    detailed = pd.DataFrame(detailed_rows)
    summary = pd.DataFrame(summary_rows)
    detailed.to_csv(args.out_detailed, index=False)
    summary.to_csv(args.out_summary, index=False)

    print(f"\nSystems missing assembled.npz (skipped): {n_missing}")
    print(f"Wrote {args.out_detailed} ({len(detailed)} rows)")
    print(f"Wrote {args.out_summary} ({len(summary)} rows)")

    if len(summary):
        shortest = summary["shortest_cutoff_tested_ps"].iloc[0]
        n_stable_short = summary["stable_at_shortest_cutoff"].sum()
        print(f"\nAt the shortest tested cutoff ({shortest:g} ps): "
              f"{n_stable_short}/{len(summary)} (system, B0) combinations are within "
              f"{args.tolerance_pct:g}% of their {args.reference_cutoff_ps:g} ps reference")
        never_stable = summary["min_stable_cutoff_ps"].isna().sum()
        if never_stable:
            print(f"WARNING: {never_stable} (system, B0) combinations never stabilize within the "
                  f"tested cutoff range, worth a manual look at those specifically.")
        print("\nDistribution of min_stable_cutoff_ps across all (system, B0) combinations:")
        print(summary["min_stable_cutoff_ps"].value_counts().sort_index())


if __name__ == "__main__":
    main()
