#!/usr/bin/env python3
"""
Step 4 (new) - Per-atom uncertainty on the powder-averaged T1, T2.

Replaces step4_per_track_uncertainty.py's role (that script fit each
track's TCF at theta=0 only). The new uncertainty, as settled for this
project: for each tracked Xe atom, fit its own TCF at every angle in the
same locked 10-point grid used by step2f_powder_averaged_relaxation.py,
powder-average that atom's own T1(theta), T2(theta) sequence (sin(theta)
weighted, on rates, exactly as step2f does for the pooled curve), giving
one powder-averaged T1, T2 per atom. The reported uncertainty is then the
mean and SEM ACROSS ATOMS of these per-atom powder-averaged values.

This is deliberately separate from the "spread of T(theta) across angle"
that step2f already reports (its T1_min/T1_max, or the full per_theta
array in its output) - that is a real, physical orientational-anisotropy
signal from the pooled (lowest-noise) curve, not a statistical error bar,
and should be read from step2f's own output, not recomputed here. This
script produces ONLY the atom-to-atom SEM, mirroring exactly how the old
pipeline drew its point estimate from the pooled fit and its uncertainty
from the per-track spread.

Takes Step 1's assembled.npz directly (needs the raw tensor to build each
track's own h0, h1, h2 at every angle), not step2_build_tcf.py's tcf.npz.

Fit bounds carried over unchanged from the old step4_per_track_uncertainty.py:
amplitudes constrained non-negative, tau capped at the fit window, for the
same reason as before, a single noisy track's unconstrained fit can find a
spurious near-zero-J(w0) component that sends T1/T2 to an unphysical value,
this is now even more important since each atom gets 10 independent noisy
fits (one per angle) instead of 1.

Usage
-----
    python step4_powder_averaged_per_track_uncertainty.py \\
        --assembled ./cnt_10_0_u23_Xe4_10ns_assembled.npz \\
        --n-angles 10 \\
        --fit-max-lag-ps 3000 \\
        --b0-fields 9.4 14.1 \\
        --out ./cnt_10_0_u23_Xe4_10ns_powder_relaxation_per_track.npz
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


def lorentzian_sum(omega, amps, taus):
    omega = np.atleast_1d(omega).astype(float)
    J = np.zeros_like(omega)
    for A, tau in zip(amps, taus):
        J += A * tau / (1.0 + (omega * tau) ** 2)
    return J


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


def powder_average_rate(theta_deg_grid, quantity_values):
    theta_rad = np.radians(np.asarray(theta_deg_grid, dtype=float))
    rates = 1.0 / np.asarray(quantity_values, dtype=float)
    weighted_rate = _trapz(rates * np.sin(theta_rad), theta_rad)
    return 1.0 / weighted_rate


# ---------------------------------------------------------------------------
# Verbatim from step4_per_track_uncertainty.py (the bounded per-track fit),
# unchanged, since the failure mode it guards against is the same or worse
# here (10 independent noisy fits per atom instead of 1).
# ---------------------------------------------------------------------------

def mono_exp(t, A, tau):
    return A * np.exp(-t / tau)


def bi_exp(t, A1, tau1, A2, tau2):
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)


def fit_track(t_ps, y, fit_max_lag_ps):
    amp_bound = 5.0 * max(np.max(np.abs(y)), 1e-10)
    p0_mono = [abs(y[0]), 2.0]
    popt_mono, _ = curve_fit(
        mono_exp, t_ps, y, p0=p0_mono, maxfev=20000,
        bounds=([0.0, 1e-3], [amp_bound, fit_max_lag_ps]),
    )
    try:
        p0_bi = [abs(y[0]) * 0.8, 1.0, abs(y[0]) * 0.2, 50.0]
        popt_bi, _ = curve_fit(
            bi_exp, t_ps, y, p0=p0_bi, maxfev=50000,
            bounds=([0.0, 1e-3, 0.0, 1e-3], [amp_bound, fit_max_lag_ps, amp_bound, fit_max_lag_ps]),
        )
        resid = y - bi_exp(t_ps, *popt_bi)
        r2 = 1.0 - np.sum(resid**2) / np.sum((y - y.mean())**2)
        return [popt_bi[0], popt_bi[2]], [popt_bi[1], popt_bi[3]], r2
    except RuntimeError:
        resid = y - mono_exp(t_ps, *popt_mono)
        r2 = 1.0 - np.sum(resid**2) / np.sum((y - y.mean())**2)
        return [popt_mono[0]], [popt_mono[1]], r2


def track_relaxation_at_theta(tensor, k, dt_fs, gamma_xe, theta_rad, fit_max_lag_ps, b0_fields):
    """Same as step2f's relaxation_at_theta, but for one track only (no
    pooling over atoms), and using the bounded per-track fit."""
    h_lab_0, h_lab_1 = lab_frame_h0_h1(tensor[:, k:k+1], gamma_xe, theta_rad)  # keep 2D, one track
    h_lab_0, h_lab_1 = h_lab_0[:, 0], h_lab_1[:, 0]
    dh0 = h_lab_0 - h_lab_0.mean()
    dh1 = h_lab_1 - h_lab_1.mean()
    tcf_h0 = fft_corr(dh0).real
    tcf_h1 = fft_corr(dh1)
    tcf_total = tcf_h0 + 2.0 * tcf_h1.real

    lag_ps = np.arange(tensor.shape[0]) * dt_fs / 1000.0
    mask = lag_ps <= fit_max_lag_ps
    t_fit_ps, y_fit = lag_ps[mask], tcf_total[mask]
    amps_ps, taus_ps, r2 = fit_track(t_fit_ps, y_fit, fit_max_lag_ps)
    taus_si = [tau * 1e-12 for tau in taus_ps]
    J0 = lorentzian_sum(0.0, amps_ps, taus_si)[0]
    b0_results = {}
    for B0 in b0_fields:
        w0 = abs(gamma_xe) * B0
        Jw0 = lorentzian_sum(w0, amps_ps, taus_si)[0]
        T1 = 1.0 / (6.0 * B0 ** 2 * Jw0)
        T2 = 1.0 / (B0 ** 2 * (4.0 * J0 + 3.0 * Jw0))
        b0_results[B0] = dict(T1=T1, T2=T2)
    return dict(r2=r2, b0_results=b0_results)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", required=True, type=Path)
    ap.add_argument("--gamma-xe", type=float, default=GAMMA_XE_DEFAULT)
    ap.add_argument("--n-angles", type=int, default=10)
    ap.add_argument("--fit-max-lag-ps", type=float, default=3000.0)
    ap.add_argument("--b0-fields", type=float, nargs="+", default=[9.4, 14.1])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    d = np.load(args.assembled)
    tensor = d["tensor"]
    dt_fs = float(d["dt_fs"])
    n_atoms = tensor.shape[1]
    print(f"Loaded {args.assembled}: {tensor.shape[0]} frames, {n_atoms} tracks, dt={dt_fs} fs")

    theta_grid_deg = list(np.linspace(0.0, 90.0, args.n_angles))
    print(f"Angle grid ({args.n_angles} points): {[round(t, 2) for t in theta_grid_deg]} deg")
    print(f"Fitting {n_atoms} tracks x {args.n_angles} angles = {n_atoms * args.n_angles} independent fits")

    # per_track_powder[B0] holds one powder-averaged T1 (and T2) per atom
    per_track_powder_T1 = {B0: [] for B0 in args.b0_fields}
    per_track_powder_T2 = {B0: [] for B0 in args.b0_fields}
    per_track_r2 = []  # mean R^2 across angles, per track, for the quality flag below

    for k in range(n_atoms):
        track_T1_by_theta = {B0: [] for B0 in args.b0_fields}
        track_T2_by_theta = {B0: [] for B0 in args.b0_fields}
        r2s_this_track = []
        for theta_deg in theta_grid_deg:
            res = track_relaxation_at_theta(tensor, k, dt_fs, args.gamma_xe, math.radians(theta_deg),
                                             args.fit_max_lag_ps, args.b0_fields)
            r2s_this_track.append(res["r2"])
            for B0 in args.b0_fields:
                track_T1_by_theta[B0].append(res["b0_results"][B0]["T1"])
                track_T2_by_theta[B0].append(res["b0_results"][B0]["T2"])
        per_track_r2.append(float(np.mean(r2s_this_track)))

        line = f"  track {k}: mean R2={per_track_r2[-1]:.4f}"
        for B0 in args.b0_fields:
            T1_pow = powder_average_rate(theta_grid_deg, track_T1_by_theta[B0])
            T2_pow = powder_average_rate(theta_grid_deg, track_T2_by_theta[B0])
            per_track_powder_T1[B0].append(T1_pow)
            per_track_powder_T2[B0].append(T2_pow)
            line += f"  T1_powder({B0:g}T)={T1_pow:.4e}s  T2_powder({B0:g}T)={T2_pow:.4e}s"
        print(line)

    print()
    print(f"{'B0 (T)':>8} {'T1 mean (s)':>14} {'T1 std (s)':>12} {'T1 SEM (s)':>12} "
          f"{'T2 mean (s)':>14} {'T2 std (s)':>12} {'T2 SEM (s)':>12}")
    summary = {}
    for B0 in args.b0_fields:
        t1 = np.array(per_track_powder_T1[B0])
        t2 = np.array(per_track_powder_T2[B0])
        sem1 = t1.std(ddof=1) / np.sqrt(n_atoms)
        sem2 = t2.std(ddof=1) / np.sqrt(n_atoms)
        print(f"{B0:8.2f} {t1.mean():14.4e} {t1.std(ddof=1):12.4e} {sem1:12.4e} "
              f"{t2.mean():14.4e} {t2.std(ddof=1):12.4e} {sem2:12.4e}")

        for label, arr in (("T1", t1), ("T2", t2)):
            assert np.all(arr > 0), (
                f"{label} at {B0:g} T came out <= 0 for at least one track's powder average, "
                f"this should be impossible given the bounded fit, investigate: {arr}"
            )
        med1, med2 = np.median(t1), np.median(t2)
        out1 = np.where(np.abs(t1 - med1) > 5 * med1)[0]
        out2 = np.where(np.abs(t2 - med2) > 5 * med2)[0]
        if len(out1) and len(out1) < n_atoms:
            print(f"  WARNING at {B0:g} T: track(s) {list(out1)} have a powder-averaged T1 more than 5x "
                  f"the median of the others ({med1:.1f} s), inspect before trusting mean/SEM.")
        if len(out2) and len(out2) < n_atoms:
            print(f"  WARNING at {B0:g} T: track(s) {list(out2)} have a powder-averaged T2 more than 5x "
                  f"the median of the others ({med2:.1f} s), inspect before trusting mean/SEM.")

        summary[B0] = dict(T1_mean=t1.mean(), T1_std=t1.std(ddof=1), T1_sem=sem1,
                            T2_mean=t2.mean(), T2_std=t2.std(ddof=1), T2_sem=sem2,
                            T1_per_track_powder=t1, T2_per_track_powder=t2)

    print(f"\nPer-track mean fit quality across angles (R^2): {[round(r, 3) for r in per_track_r2]}")
    if min(per_track_r2) < 0.7:
        print("NOTE: at least one track fits noticeably worse than the others on average across angles, "
              "worth a look before trusting the pooled uncertainty.")

    np.savez_compressed(args.out, b0_fields=args.b0_fields, theta_grid_deg=theta_grid_deg,
                         summary=summary, per_track_r2=per_track_r2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
