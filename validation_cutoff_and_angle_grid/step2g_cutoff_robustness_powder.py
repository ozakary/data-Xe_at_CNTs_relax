#!/usr/bin/env python3
"""
Step 2g - Cutoff robustness for the powder-averaged methodology.

Same role as step5_cutoff_sensitivity.py / step6_check_cutoff_robustness.py,
but for the new question: is fit-max-lag-ps=3000 still the right choice now
that fitting happens at 10 angles per system, not just theta=0. The old
validation only ever checked stability at theta=0, it says nothing about
whether every angle's lab-frame TCF (a different weighted mixture of the
tube-frame h0, h1, h2 correlation functions at each theta) decays on the
same timescale, or needs a different cutoff to resolve.

Efficiency note: the expensive part is building each angle's lab-frame TCF
(the Wigner rotation combination, then FFT autocorrelation), not the
exponential fit itself. So each angle's TCF is built ONCE, out to the full
trajectory length, then refit at every requested cutoff, rather than
rebuilding the TCF from scratch per cutoff (which is what repeated calls to
step2f_powder_averaged_relaxation.py would do). The reference cutoff
(--reference-cutoff-ps, default 3000, matching the production value) is
computed fresh here too, self-contained, rather than read from an existing
_powder_relaxation.npz, so this script has no dependency on that file
existing or matching exactly.

Usage
-----
    python step2g_cutoff_robustness_powder.py \\
        --assembled ./cnt_10_0_u23_Xe4_10ns_assembled.npz \\
        --n-angles 10 \\
        --cutoffs-ps 20 50 100 200 300 500 750 1000 1500 2000 3000 \\
        --reference-cutoff-ps 3000 \\
        --b0-fields 9.4 14.1 \\
        --out ./cnt_10_0_u23_Xe4_10ns_powder_cutoff_robustness.npz \\
        --plot ./cnt_10_0_u23_Xe4_10ns_powder_cutoff_robustness.png
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
# Verbatim from step2f_powder_averaged_relaxation.py, copied rather than
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
    # Non-negative amplitudes, same fix and same justification as
    # step2f_powder_averaged_relaxation.py's fit_pooled, applied here too
    # since this script refits the same kind of pooled TCF at many cutoffs,
    # the shorter the cutoff the noisier the fit, so if anything this
    # matters MORE here than in step2f's single (long) cutoff case.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", required=True, type=Path)
    ap.add_argument("--gamma-xe", type=float, default=GAMMA_XE_DEFAULT)
    ap.add_argument("--n-angles", type=int, default=10)
    ap.add_argument("--cutoffs-ps", type=float, nargs="+",
                     default=[20, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000])
    ap.add_argument("--reference-cutoff-ps", type=float, default=3000.0,
                     help="must be one of --cutoffs-ps; treated as the trustworthy reference, "
                          "matching the production fit-max-lag-ps")
    ap.add_argument("--b0-fields", type=float, nargs="+", default=[9.4, 14.1])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    if args.reference_cutoff_ps not in args.cutoffs_ps:
        args.cutoffs_ps = sorted(set(args.cutoffs_ps) | {args.reference_cutoff_ps})
        print(f"Note: added --reference-cutoff-ps to --cutoffs-ps, now: {args.cutoffs_ps}")

    d = np.load(args.assembled)
    tensor = d["tensor"]
    dt_fs = float(d["dt_fs"])
    print(f"Loaded {args.assembled}: {tensor.shape[0]} frames, {tensor.shape[1]} tracks, dt={dt_fs} fs")

    theta_grid_deg = list(np.linspace(0.0, 90.0, args.n_angles))
    lag_ps_full = np.arange(tensor.shape[0]) * dt_fs / 1000.0
    max_cutoff = max(args.cutoffs_ps)

    # --- Build each angle's TCF ONCE, out to the longest requested cutoff ---
    print(f"Building {args.n_angles} angle TCFs (the expensive step, done once each)...")
    tcf_by_theta = {}
    for theta_deg in theta_grid_deg:
        h_lab_0, h_lab_1 = lab_frame_h0_h1(tensor, args.gamma_xe, math.radians(theta_deg))
        dh0 = h_lab_0 - h_lab_0.mean(axis=0)[None, :]
        dh1 = h_lab_1 - h_lab_1.mean(axis=0)[None, :]
        n_frames, n_atoms = tensor.shape[0], tensor.shape[1]
        tcf_h0 = np.empty((n_frames, n_atoms))
        tcf_h1 = np.empty((n_frames, n_atoms), dtype=complex)
        for k in range(n_atoms):
            tcf_h0[:, k] = fft_corr(dh0[:, k]).real
            tcf_h1[:, k] = fft_corr(dh1[:, k])
        tcf_by_theta[theta_deg] = tcf_h0.mean(axis=1) + 2.0 * tcf_h1.mean(axis=1).real
        print(f"  theta={theta_deg:6.2f} deg done")

    # --- Refit each angle's (already-built) TCF at every cutoff ---
    print(f"\nRefitting at {len(args.cutoffs_ps)} cutoffs (cheap, reusing the TCFs above)...")
    powder_by_cutoff = {}
    per_theta_by_cutoff = {}
    for cutoff in sorted(args.cutoffs_ps):
        mask = lag_ps_full <= cutoff
        if mask.sum() < 20:
            print(f"  cutoff={cutoff:8.1f} ps  skipped (not enough points)")
            continue
        per_theta_results = {}
        for theta_deg in theta_grid_deg:
            y_fit = tcf_by_theta[theta_deg][mask]
            amps_ps, taus_ps, r2 = fit_pooled(lag_ps_full[mask], y_fit)
            per_theta_results[theta_deg] = dict(r2=r2, b0=relax_from_fit(amps_ps, taus_ps, args.gamma_xe, args.b0_fields))
        per_theta_by_cutoff[cutoff] = per_theta_results

        powder = {}
        for B0 in args.b0_fields:
            T1s = [per_theta_results[t]["b0"][B0]["T1"] for t in theta_grid_deg]
            T2s = [per_theta_results[t]["b0"][B0]["T2"] for t in theta_grid_deg]
            powder[B0] = dict(T1=powder_average_rate(theta_grid_deg, T1s),
                               T2=powder_average_rate(theta_grid_deg, T2s))
        powder_by_cutoff[cutoff] = powder
        line = f"  cutoff={cutoff:8.1f} ps"
        for B0 in args.b0_fields:
            line += f"  T1({B0:g}T)={powder[B0]['T1']:.4e}s  T2({B0:g}T)={powder[B0]['T2']:.4e}s"
        print(line)

    ref = powder_by_cutoff[args.reference_cutoff_ps]
    print(f"\nPercent difference from the {args.reference_cutoff_ps:g} ps reference:")
    rows = []
    for cutoff in sorted(powder_by_cutoff):
        pw = powder_by_cutoff[cutoff]
        row = dict(cutoff_ps=cutoff)
        line = f"  cutoff={cutoff:8.1f} ps"
        for B0 in args.b0_fields:
            t1_pct = 100.0 * (pw[B0]["T1"] - ref[B0]["T1"]) / ref[B0]["T1"]
            t2_pct = 100.0 * (pw[B0]["T2"] - ref[B0]["T2"]) / ref[B0]["T2"]
            row[f"T1_pct_diff_{B0}"] = t1_pct
            row[f"T2_pct_diff_{B0}"] = t2_pct
            line += f"  T1_diff({B0:g}T)={t1_pct:+7.2f}%  T2_diff({B0:g}T)={t2_pct:+7.2f}%"
        print(line)
        rows.append(row)

    np.savez_compressed(args.out, theta_grid_deg=theta_grid_deg, cutoffs_ps=args.cutoffs_ps,
                         b0_fields=args.b0_fields, powder_by_cutoff=powder_by_cutoff,
                         per_theta_by_cutoff=per_theta_by_cutoff, reference_cutoff_ps=args.reference_cutoff_ps)
    print(f"\nWrote {args.out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(args.b0_fields), figsize=(5.5 * len(args.b0_fields), 4), squeeze=False)
        cutoffs_sorted = sorted(powder_by_cutoff)
        for i, B0 in enumerate(args.b0_fields):
            ax = axes[0, i]
            t1_pct = [row[f"T1_pct_diff_{B0}"] for row in rows]
            t2_pct = [row[f"T2_pct_diff_{B0}"] for row in rows]
            ax.plot(cutoffs_sorted, t1_pct, "o-", label="T1")
            ax.plot(cutoffs_sorted, t2_pct, "s-", label="T2")
            ax.axhspan(-10, 10, color="gray", alpha=0.15)
            ax.axhline(0, color="0.5", lw=0.6)
            ax.set_xscale("log")
            ax.set_xlabel("fit cutoff (ps)")
            ax.set_ylabel(f"%% deviation from {args.reference_cutoff_ps:g} ps reference")
            ax.set_title(f"B0 = {B0} T")
            ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
