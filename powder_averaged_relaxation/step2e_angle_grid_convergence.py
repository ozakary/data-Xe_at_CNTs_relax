#!/usr/bin/env python3
"""
Step 2e - Angle-grid convergence test.

Same role as step5_cutoff_sensitivity.py, but for the new question: how many
theta points, and at what spacing, does the powder-averaged T1/T2 actually
need before it stops changing. Runs a fine reference grid and several
coarser candidate grids on one system, reports the percent difference of
each candidate's powder average from the fine-grid reference, so the grid
size used in production (Steps 2d/4-onward) is a demonstrated, stable
choice rather than an unexamined pick, exactly the same logic already
applied to the fit-window choice.

Powder average, as settled for this project: theta ranges over [0, 90]
degrees (by up-down symmetry of the tube, this covers the full range), each
angle weighted by its share of solid angle, sin(theta) dtheta, and the
averaging is performed on the RATES 1/T1, 1/T2 (matching Redfield theory,
rates are the additive quantity), not on the times themselves:

    <1/T1> = integral_0^(pi/2) [1/T1(theta)] sin(theta) dtheta
    T1_powder = 1 / <1/T1>

Implemented as trapezoidal quadrature directly on whatever theta grid is
given (not assumed evenly spaced), which converges to the true integral as
the grid is refined, exactly what this test demonstrates empirically.

Reuses step2d_tilt_angle_relaxation.py's validated tube-frame construction
and Wigner-rotation combination (copied, not imported, so this script has
no dependency on that file and cannot silently diverge if it changes).

Usage
-----
    python step2e_angle_grid_convergence.py \\
        --assembled ./cnt_10_0_u23_Xe4_10ns_assembled.npz \\
        --reference-n-angles 46 \\
        --candidate-grids "0,90" "0,45,90" "0,30,60,90" "0,22.5,45,67.5,90" \\
                          "0,15,30,45,60,75,90" "0,10,20,30,40,50,60,70,80,90" \\
        --fit-max-lag-ps 3000 \\
        --b0-fields 9.4 14.1 \\
        --out ./cnt_10_0_u23_Xe4_10ns_angle_grid_convergence.npz \\
        --plot ./cnt_10_0_u23_Xe4_10ns_angle_grid_convergence.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

GAMMA_XE_DEFAULT = -7.451956e7  # rad s^-1 T^-1, 129Xe gyromagnetic ratio, matches step2_build_tcf.py
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Verbatim from step2d_tilt_angle_relaxation.py, copied rather than imported.
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


def fit_track(t_ps, y):
    p0_mono = [y[0], 2.0]
    popt_mono, _ = curve_fit(mono_exp, t_ps, y, p0=p0_mono, maxfev=20000)
    r2_mono = fit_quality(t_ps, y, mono_exp, popt_mono)
    try:
        p0_bi = [y[0] * 0.8, 1.0, y[0] * 0.2, 50.0]
        popt_bi, _ = curve_fit(bi_exp, t_ps, y, p0=p0_bi, maxfev=50000,
                                bounds=([-np.inf, 1e-3, -np.inf, 1e-3], [np.inf, np.inf, np.inf, np.inf]))
        r2_bi = fit_quality(t_ps, y, bi_exp, popt_bi)
        return [popt_bi[0], popt_bi[2]], [popt_bi[1], popt_bi[3]], r2_bi
    except RuntimeError:
        return [popt_mono[0]], [popt_mono[1]], r2_mono


def relaxation_at_theta(tensor, dt_fs, gamma_xe, theta_rad, fit_max_lag_ps, b0_fields):
    n_frames, n_atoms = tensor.shape[0], tensor.shape[1]
    h_lab_0, h_lab_1 = lab_frame_h0_h1(tensor, gamma_xe, theta_rad)
    dh0 = h_lab_0 - h_lab_0.mean(axis=0)[None, :]
    dh1 = h_lab_1 - h_lab_1.mean(axis=0)[None, :]
    tcf_h0 = np.empty((n_frames, n_atoms))
    tcf_h1 = np.empty((n_frames, n_atoms), dtype=complex)
    for k in range(n_atoms):
        tcf_h0[:, k] = fft_corr(dh0[:, k]).real
        tcf_h1[:, k] = fft_corr(dh1[:, k])
    tcf_total = tcf_h0.mean(axis=1) + 2.0 * tcf_h1.mean(axis=1).real
    lag_ps = np.arange(n_frames) * dt_fs / 1000.0
    mask = lag_ps <= fit_max_lag_ps
    t_fit_ps, y_fit = lag_ps[mask], tcf_total[mask]
    amps_ps, taus_ps, r2 = fit_track(t_fit_ps, y_fit)
    taus_si = [tau * 1e-12 for tau in taus_ps]
    J0 = lorentzian_sum(0.0, amps_ps, taus_si)[0]
    results = {}
    for B0 in b0_fields:
        w0 = abs(gamma_xe) * B0
        Jw0 = lorentzian_sum(w0, amps_ps, taus_si)[0]
        T1 = 1.0 / (6.0 * B0 ** 2 * Jw0) if Jw0 > 0 else float("nan")
        T2 = 1.0 / (B0 ** 2 * (4.0 * J0 + 3.0 * Jw0)) if (4.0 * J0 + 3.0 * Jw0) > 0 else float("nan")
        results[B0] = dict(T1=T1, T2=T2)
    return results


# ---------------------------------------------------------------------------
# New: powder averaging on rates, sin(theta)-weighted, plus the grid-density
# convergence comparison.
# ---------------------------------------------------------------------------

def powder_average_rate(theta_deg_grid, quantity_values):
    """theta_deg_grid: degrees, must span [0, 90]. quantity_values: T1 or T2
    at each grid point. Returns the powder-averaged TIME (reciprocal of the
    sin(theta)-weighted average RATE), via trapezoidal quadrature on the
    given grid, whatever its spacing."""
    theta_rad = np.radians(np.asarray(theta_deg_grid, dtype=float))
    rates = 1.0 / np.asarray(quantity_values, dtype=float)
    weighted_rate = _trapz(rates * np.sin(theta_rad), theta_rad)
    return 1.0 / weighted_rate


def run_grid(tensor, dt_fs, gamma_xe, theta_deg_list, fit_max_lag_ps, b0_fields):
    per_theta = {}
    for theta_deg in theta_deg_list:
        per_theta[theta_deg] = relaxation_at_theta(tensor, dt_fs, gamma_xe, math.radians(theta_deg),
                                                     fit_max_lag_ps, b0_fields)
    powder = {}
    for B0 in b0_fields:
        T1s = [per_theta[t][B0]["T1"] for t in theta_deg_list]
        T2s = [per_theta[t][B0]["T2"] for t in theta_deg_list]
        powder[B0] = dict(T1=powder_average_rate(theta_deg_list, T1s),
                           T2=powder_average_rate(theta_deg_list, T2s))
    return per_theta, powder


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", required=True, type=Path)
    ap.add_argument("--gamma-xe", type=float, default=GAMMA_XE_DEFAULT)
    ap.add_argument("--reference-n-angles", type=int, default=46,
                     help="number of evenly-spaced points from 0 to 90 deg used as the fine reference grid")
    ap.add_argument("--candidate-grids", type=str, nargs="+", required=True,
                     help="each as a comma-separated list of angles in degrees, e.g. '0,45,90'")
    ap.add_argument("--fit-max-lag-ps", type=float, default=3000.0)
    ap.add_argument("--b0-fields", type=float, nargs="+", default=[9.4, 14.1])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.assembled)
    tensor = d["tensor"]
    dt_fs = float(d["dt_fs"])
    print(f"Loaded {args.assembled}: {tensor.shape[0]} frames, {tensor.shape[1]} tracks, dt={dt_fs} fs")

    print(f"\nReference grid: {args.reference_n_angles} evenly-spaced points, 0-90 deg")
    ref_grid = list(np.linspace(0.0, 90.0, args.reference_n_angles))
    _, ref_powder = run_grid(tensor, dt_fs, args.gamma_xe, ref_grid, args.fit_max_lag_ps, args.b0_fields)
    for B0 in args.b0_fields:
        print(f"  B0={B0:g}T  T1_powder={ref_powder[B0]['T1']:.4e}s  T2_powder={ref_powder[B0]['T2']:.4e}s")

    all_candidates = []
    print(f"\n{'grid (n pts)':>14} {'B0':>6} {'T1_powder':>12} {'T1 % diff':>10} {'T2_powder':>12} {'T2 % diff':>10}")
    for grid_str in args.candidate_grids:
        grid = [float(x) for x in grid_str.split(",")]
        _, powder = run_grid(tensor, dt_fs, args.gamma_xe, grid, args.fit_max_lag_ps, args.b0_fields)
        for B0 in args.b0_fields:
            t1_pct = 100.0 * (powder[B0]["T1"] - ref_powder[B0]["T1"]) / ref_powder[B0]["T1"]
            t2_pct = 100.0 * (powder[B0]["T2"] - ref_powder[B0]["T2"]) / ref_powder[B0]["T2"]
            print(f"{grid_str + f' ({len(grid)})':>14} {B0:>6g} {powder[B0]['T1']:12.4e} {t1_pct:10.2f} "
                  f"{powder[B0]['T2']:12.4e} {t2_pct:10.2f}")
            all_candidates.append(dict(grid=grid, n_points=len(grid), B0=B0,
                                        T1_powder=powder[B0]["T1"], T2_powder=powder[B0]["T2"],
                                        T1_pct_diff=t1_pct, T2_pct_diff=t2_pct))

    np.savez_compressed(args.out, reference_grid=ref_grid, reference_powder=ref_powder,
                         candidates=all_candidates, b0_fields=args.b0_fields)
    print(f"\nWrote {args.out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(args.b0_fields), figsize=(5.5 * len(args.b0_fields), 4), squeeze=False)
        for i, B0 in enumerate(args.b0_fields):
            ax = axes[0, i]
            rows = [c for c in all_candidates if c["B0"] == B0]
            ns = [r["n_points"] for r in rows]
            ax.plot(ns, [r["T1_pct_diff"] for r in rows], "o-", label="T1")
            ax.plot(ns, [r["T2_pct_diff"] for r in rows], "s-", label="T2")
            ax.axhspan(-5, 5, color="gray", alpha=0.15)
            ax.axhline(0, color="0.5", lw=0.6)
            ax.set_xlabel("number of angle grid points")
            ax.set_ylabel(f"% deviation from {args.reference_n_angles}-point reference")
            ax.set_title(f"B0 = {B0} T")
            ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
