#!/usr/bin/env python3
"""
Prints ready-to-paste LaTeX rows for Tables S1-S3 (TCF fit parameters:
A1, A2, tau1, tau2, R^2, <h0>), one block per series (temperature,
diameter, loading), in the same row order used in those tables, now with
one row per (system, theta), at theta = 0, 40, 90 degrees (the same
reduced set used in plot_tcf_fit_diagnostics.py, kept small for the same
reason, the full 10-point grid would make these tables unwieldy).

Rebuilds the TCF from Step 1's assembled.npz at each requested angle (the
same Wigner-rotation combination used throughout the powder-averaged
methodology, copied verbatim from plot_tcf_fit_diagnostics.py) and fits
both mono- and bi-exponential models explicitly, reporting whichever one
step2f_powder_averaged_relaxation.py's own convention would report (bi-
exponential if it converged, else the mono-exponential fallback), R^2 for
that same chosen fit. This cannot be read directly from step2f's saved
output, that file only keeps the winning model's parameters, not both, and
never saved <h_lab,0> (the lab-frame axial mean) at any angle, only the
tube-frame h0 was ever persisted (Step 2's per-track means, theta=0 only).

<h0> here is therefore <h_lab,0> at the given theta, the plain (signed)
mean of the lab-frame h0(t) before mean-subtraction, the direct
theta-dependent generalization of the old table's tube-frame quantity
(which is exactly the theta=0 row's value).

Usage
-----
    python print_tcf_fit_params_table.py \\
        --manifest ./run_manifest.csv \\
        --results-dir ./results \\
        --theta-deg 0 40 90
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

GAMMA_XE_DEFAULT = -7.451956e7  # rad s^-1 T^-1, 129Xe gyromagnetic ratio, matches step2_build_tcf.py


# ---------------------------------------------------------------------------
# Verbatim from plot_tcf_fit_diagnostics.py, copied rather than imported so
# this script has no dependency on that file.
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


def build_tcf_at_theta(assembled_path: Path, gamma_xe: float, theta_rad: float):
    """Returns (lag_ps, tcf, h_lab_0_mean_per_track)."""
    d = np.load(assembled_path)
    tensor = d["tensor"]
    dt_fs = float(d["dt_fs"])
    n_frames, n_atoms = tensor.shape[0], tensor.shape[1]
    h_lab_0, h_lab_1 = lab_frame_h0_h1(tensor, gamma_xe, theta_rad)
    h_lab_0_mean = h_lab_0.mean(axis=0)
    dh0 = h_lab_0 - h_lab_0_mean[None, :]
    dh1 = h_lab_1 - h_lab_1.mean(axis=0)[None, :]
    tcf_h0 = np.empty((n_frames, n_atoms))
    tcf_h1 = np.empty((n_frames, n_atoms), dtype=complex)
    for k in range(n_atoms):
        tcf_h0[:, k] = fft_corr(dh0[:, k]).real
        tcf_h1[:, k] = fft_corr(dh1[:, k])
    tcf = tcf_h0.mean(axis=1) + 2.0 * tcf_h1.mean(axis=1).real
    lag_ps = np.arange(n_frames) * dt_fs / 1000.0
    return lag_ps, tcf, h_lab_0_mean


def mono_exp(t, A, tau):
    return A * np.exp(-t / tau)


def bi_exp(t, A1, tau1, A2, tau2):
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)


def fit_quality(t, y, model, popt):
    resid = y - model(t, *popt)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def fit_reported(t_ps, y, fit_max_lag_ps):
    """Same non-negative-amplitude bounds as step2f_powder_averaged_relaxation.py's
    fit_pooled, and the same reporting convention, bi-exponential if it
    converged, else the mono-exponential fallback, whichever step2f itself
    would have reported for this (system, theta)."""
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


CASE_CONFIG = {
    1: dict(var_label="$T$ / K", subcase_order=["low", "high"],
            subcase_display={"low": "LL", "high": "HL"}),
    2: dict(var_label="Tube", subcase_order=["low", "high"],
            subcase_display={"low": "LL", "high": "HL"}),
    3: dict(var_label="Loading / \\%", subcase_order=["cnt_10_0", "cnt_30_0"],
            subcase_display={"cnt_10_0": "(10,0)", "cnt_30_0": "(30,0)"}),
}


def fmt(v, sig=4):
    return f"{v:.{sig}g}"


def get_rows(run_id: str, results_dir: Path, var_display: str, loading_display: str,
             theta_grid_deg, gamma_xe: float, fit_max_lag_ps: float):
    sys_dir = results_dir / run_id
    assembled_path = sys_dir / f"{run_id}_assembled.npz"
    if not assembled_path.exists():
        print(f"  % MISSING DATA for {run_id}")
        return []

    lines = []
    for theta_deg in theta_grid_deg:
        lag_ps, tcf, h_lab_0_mean = build_tcf_at_theta(assembled_path, gamma_xe, math.radians(theta_deg))
        h0_mean = float(np.mean(h_lab_0_mean.real))

        mask = lag_ps <= fit_max_lag_ps
        t, y = lag_ps[mask], tcf[mask]
        amps, taus, r2 = fit_reported(t, y, fit_max_lag_ps)

        if len(amps) == 2:
            A1, tau1, A2, tau2 = amps[0], taus[0], amps[1], taus[1]
        else:
            A1, tau1 = amps[0], taus[0]
            A2, tau2 = float("nan"), float("nan")

        a2_str = fmt(A2) if np.isfinite(A2) else "--"
        tau2_str = fmt(tau2) if np.isfinite(tau2) else "--"
        theta_str = f"${theta_deg:g}^\\circ$"
        lines.append(f"    {loading_display} & {var_display} & {theta_str} & {fmt(A1)} & {fmt(tau1)} & "
                      f"{a2_str} & {tau2_str} & {r2:.3f} & {fmt(h0_mean)} \\\\")
    return lines


def print_case(manifest: pd.DataFrame, case: int, results_dir: Path,
                theta_grid_deg, gamma_xe: float, fit_max_lag_ps: float):
    config = CASE_CONFIG[case]
    cdf = manifest[manifest["case"] == case]
    print(f"% --- Case {case} ({config['var_label']}) ---")
    for subcase in config["subcase_order"]:
        sub = cdf[cdf["subcase"] == subcase].sort_values("variable_value")
        loading_display = config["subcase_display"][subcase]
        for _, row in sub.iterrows():
            if case == 2:
                var_display = row["chirality"]  # e.g. "(10,0)"
            elif case == 3:
                var_display = f"{row['variable_value']:.1f}"
            else:
                var_display = f"{row['variable_value']:g}"
            for line in get_rows(row["run_id"], results_dir, var_display, loading_display,
                                  theta_grid_deg, gamma_xe, fit_max_lag_ps):
                print(line)
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--theta-deg", type=float, nargs="+", default=[0, 40, 90],
                     help="matches plot_tcf_fit_diagnostics.py's reduced angle set")
    ap.add_argument("--gamma-xe", type=float, default=GAMMA_XE_DEFAULT)
    ap.add_argument("--fit-max-lag-ps", type=float, default=3000.0)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    for case in [1, 2, 3]:
        print_case(manifest, case, args.results_dir, args.theta_deg, args.gamma_xe, args.fit_max_lag_ps)


if __name__ == "__main__":
    main()
