#!/usr/bin/env python3
"""
Step 2f - Powder-averaged CSA relaxation, the new canonical "give me T1, T2
for this system" calculation, replacing step3_spectral_density_and_relaxation.py's
role in the pipeline (that script assumed theta=0 only, the SWCNT axis
exactly parallel to B0, and is kept only for reference/comparison against
this new result).

Takes Step 1's assembled.npz directly (the raw shielding tensor), not
Step 2's tcf.npz, since the tube-frame h2(t) component needed at every
angle other than 0 is not part of that file.

For each angle in the grid (locked to 10 evenly-spaced points, 0-90 deg,
following the convergence test in step2e_angle_grid_convergence.py, run on
both (10,0) and (30,0), which showed this grid holds both systems under
0.5% of a 46-point fine reference):

    1. Build h_lab,0(t), h_lab,1(t) at that angle via the Wigner rotation
       of the tube-frame h0(t), h1(t), h2(t) (see step2d_tilt_angle_relaxation.py
       for the full derivation and citation, R.Y. Dong, Nuclear Magnetic
       Resonance of Liquid Crystals, Springer 1994, Ch. 5 Eq. 5.31 and
       Ch. 7.2.2 Eqs. 7.58-7.59).
    2. Pool (average) over tracked Xe atoms, mean-subtract, FFT-autocorrelate,
       exactly mirroring step2_build_tcf.py's approach, just applied to the
       rotated h_lab,0(t), h_lab,1(t) instead of the theta=0 h0(t), h1(t).
    3. Fit mono/bi-exponential, get T1(theta), T2(theta) at each B0 field.

The reported T1, T2 for the system are then the powder average over this
grid, weighted by sin(theta) (each angle's share of solid angle), performed
on the RATES 1/T1, 1/T2 (Redfield theory's own additive quantity), not on
the times:

    <1/T1> = integral_0^(pi/2) [1/T1(theta)] sin(theta) dtheta
    T1_powder = 1 / <1/T1>

Every per-angle T1(theta), T2(theta), and fit parameter is also saved, not
just the powder average, these feed the angular-dependence figures planned
for the ESI.

Usage
-----
    python step2f_powder_averaged_relaxation.py \\
        --assembled ./cnt_10_0_u23_Xe4_10ns_assembled.npz \\
        --n-angles 10 \\
        --fit-max-lag-ps 3000 \\
        --b0-fields 9.4 14.1 \\
        --out ./cnt_10_0_u23_Xe4_10ns_powder_relaxation.npz \\
        --plot ./cnt_10_0_u23_Xe4_10ns_powder_relaxation.png
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
# Verbatim from step2d_tilt_angle_relaxation.py / step2e_angle_grid_convergence.py,
# copied rather than imported so this script has no dependency on those
# files and cannot silently diverge if they change later.
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
    # Amplitudes constrained non-negative, matching step4's own established
    # fix for exactly this failure mode (found there first, for a per-track
    # fit; the pooled fit here was never bounded, because it never hit this
    # case at theta=0 alone across all 38 systems, this changed once
    # fitting happens at 10 angles per system instead of 1). With tau > 0
    # and every amplitude >= 0, J(omega) = sum_i A_i*tau_i/(1+(omega*tau_i)^2)
    # is a sum of strictly non-negative terms for any omega, so
    # 4*J(0)+3*J(w0) cannot come out <= 0, which is what silently produced
    # a NaN T2 (and poisoned that system's entire powder average) before
    # this fix, see the conversation record.
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
    amps_ps, taus_ps, r2 = fit_pooled(t_fit_ps, y_fit)
    taus_si = [tau * 1e-12 for tau in taus_ps]
    J0 = lorentzian_sum(0.0, amps_ps, taus_si)[0]
    b0_results = {}
    for B0 in b0_fields:
        w0 = abs(gamma_xe) * B0
        Jw0 = lorentzian_sum(w0, amps_ps, taus_si)[0]
        T1 = 1.0 / (6.0 * B0 ** 2 * Jw0) if Jw0 > 0 else float("nan")
        T2 = 1.0 / (B0 ** 2 * (4.0 * J0 + 3.0 * Jw0)) if (4.0 * J0 + 3.0 * Jw0) > 0 else float("nan")
        b0_results[B0] = dict(T1=T1, T2=T2)
    return dict(amps_ps=amps_ps, taus_ps=taus_ps, r2=r2, tcf0=float(tcf_total[0]), b0_results=b0_results)


def powder_average_rate(theta_deg_grid, quantity_values):
    """Powder average of a relaxation TIME over theta in [0,90] deg, weighted
    by sin(theta), performed on the RATE 1/quantity, via trapezoidal
    quadrature on the given grid. Verified against an analytic test case
    in the conversation record before this script was written."""
    theta_rad = np.radians(np.asarray(theta_deg_grid, dtype=float))
    rates = 1.0 / np.asarray(quantity_values, dtype=float)
    weighted_rate = _trapz(rates * np.sin(theta_rad), theta_rad)
    return 1.0 / weighted_rate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", required=True, type=Path)
    ap.add_argument("--gamma-xe", type=float, default=GAMMA_XE_DEFAULT)
    ap.add_argument("--n-angles", type=int, default=10,
                     help="evenly-spaced points from 0 to 90 deg, locked to 10 after the convergence test "
                          "in step2e_angle_grid_convergence.py (both (10,0) and (30,0) under 0.5%% of a "
                          "46-point fine reference at this grid size)")
    ap.add_argument("--fit-max-lag-ps", type=float, default=3000.0)
    ap.add_argument("--b0-fields", type=float, nargs="+", default=[9.4, 14.1])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.assembled)
    tensor = d["tensor"]
    dt_fs = float(d["dt_fs"])
    print(f"Loaded {args.assembled}: {tensor.shape[0]} frames, {tensor.shape[1]} tracks, dt={dt_fs} fs")

    theta_grid_deg = list(np.linspace(0.0, 90.0, args.n_angles))
    print(f"Angle grid ({args.n_angles} points): {[round(t, 2) for t in theta_grid_deg]} deg")

    per_theta = {}
    for theta_deg in theta_grid_deg:
        res = relaxation_at_theta(tensor, dt_fs, args.gamma_xe, math.radians(theta_deg),
                                   args.fit_max_lag_ps, args.b0_fields)
        per_theta[theta_deg] = res
        line = f"  theta={theta_deg:6.2f} deg  R2={res['r2']:.4f}"
        for B0, r in res["b0_results"].items():
            line += f"  T1({B0:g}T)={r['T1']:.4e}s  T2({B0:g}T)={r['T2']:.4e}s"
        print(line)

    print("\nPowder average (sin(theta)-weighted, on rates):")
    powder = {}
    for B0 in args.b0_fields:
        T1s = [per_theta[t]["b0_results"][B0]["T1"] for t in theta_grid_deg]
        T2s = [per_theta[t]["b0_results"][B0]["T2"] for t in theta_grid_deg]
        T1_powder = powder_average_rate(theta_grid_deg, T1s)
        T2_powder = powder_average_rate(theta_grid_deg, T2s)
        powder[B0] = dict(T1=T1_powder, T2=T2_powder,
                           T1_min=min(T1s), T1_max=max(T1s), T2_min=min(T2s), T2_max=max(T2s))
        print(f"  B0={B0:g}T  T1_powder={T1_powder:.4e}s (range {min(T1s):.4e}-{max(T1s):.4e}s across theta)  "
              f"T2_powder={T2_powder:.4e}s (range {min(T2s):.4e}-{max(T2s):.4e}s across theta)")

    np.savez_compressed(args.out, theta_grid_deg=theta_grid_deg, b0_fields=args.b0_fields,
                         per_theta=per_theta, powder=powder, gamma_xe=args.gamma_xe,
                         fit_max_lag_ps=args.fit_max_lag_ps)
    print(f"\nWrote {args.out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(args.b0_fields), figsize=(5.5 * len(args.b0_fields), 4), squeeze=False)
        for i, B0 in enumerate(args.b0_fields):
            ax = axes[0, i]
            T1s = [per_theta[t]["b0_results"][B0]["T1"] for t in theta_grid_deg]
            T2s = [per_theta[t]["b0_results"][B0]["T2"] for t in theta_grid_deg]
            ax.plot(theta_grid_deg, T1s, "o-", label="$T_1(\\theta)$")
            ax.plot(theta_grid_deg, T2s, "s-", label="$T_2(\\theta)$")
            ax.axhline(powder[B0]["T1"], color="C0", ls="--", lw=1, label="$T_1$ powder avg")
            ax.axhline(powder[B0]["T2"], color="C1", ls="--", lw=1, label="$T_2$ powder avg")
            ax.set_xlabel("theta (deg)")
            ax.set_ylabel("time (s)")
            ax.set_title(f"B0 = {B0} T")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
