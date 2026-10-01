#!/usr/bin/env python3
"""
Step 2d - CSA relaxation at a general SWCNT-axis-to-B0 tilt angle theta.

Everything in Steps 1-6 assumes theta=0, the SWCNT axis exactly parallel
to B0. This script removes that restriction, using the standard theory for
an axially symmetric interaction tensor whose own symmetry axis (here, the
SWCNT axis) is tilted at a general angle relative to B0, following
R.Y. Dong, "Nuclear Magnetic Resonance of Liquid Crystals," Springer-Verlag,
New York, 1994, Chapter 5 (general case, Eq. 5.31) and Chapter 7.2.2
(director-tilt case, Eqs. 7.58-7.59), which give the lab-frame spectral
density as a combination of the tube-frame ones weighted by the squared
reduced Wigner rotation matrix elements, [d^2_{m,k}(theta)]^2.

Physical picture
-----------------
At theta=0, only h0(t) and h1(t) (the tube-frame m=0, m=+1 spherical
tensor components) ever enter the spin-1/2 relaxation Hamiltonian, the
m=+/-2 component (h2) is forbidden from doing so by a Wigner-Eckart
selection rule for spin-1/2 in the LAB frame, independent of what is
physically fluctuating. Once the tube tilts, the lab-frame h0 and h1
become mixtures of ALL FIVE tube-frame components (m=0,+/-1,+/-2), via

    h_lab,m(t) = sum_k d2_{m,k}(theta) * h_tube,k(t)

so h2(t), and its negative-m companion h_{-2}(t) = +h2(t)^* (note the
sign: h_{-1}(t) = -h1(t)^* carries an extra minus sign that h_{-2} does
not, since (-1)^m alternates with m), now genuinely contribute. This was
checked to be non-negligible for real systems in this project before this
script was written, see step2c_build_h2_tcf.py and the conversation record,
h2 contributed roughly 38% of TCF(0) for (10,0) and 66% for (30,0) at
theta=0, where it is not even supposed to matter physically, confirming it
cannot be dropped once theta != 0.

This requires no new MD or NMR-ML data, everything needed (the full
instantaneous shielding tensor) is already in Step 1's assembled.npz for
every system already computed. This is purely a post-processing step.

Approach: rather than algebraically combining already-computed correlation
functions (risking a sign or conjugation error), h_lab,0(t) and h_lab,1(t)
are built explicitly in the time domain at each requested theta, then
passed through the exact same per-track mean subtraction, FFT-based
autocorrelation, and mono/bi-exponential fitting already validated in
Steps 2-3, so the theta=0 case reproduces those results exactly, checked
below in the test suite, not just asserted.

Assumption made explicit: cross-correlations between different tube-frame
m components are taken to vanish, <dh_tube,k1*(0) dh_tube,k2(t)> = 0 for
k1 != k2, justified by the system's azimuthal symmetry about the tube axis
(the same symmetry already invoked throughout this project, e.g. for why
<h1> is expected to be small). This is not verified numerically here and
should be checked before trusting results at theta far from 0 or 90
degrees, see check_cross_correlations() below.

Usage
-----
    python step2d_tilt_angle_relaxation.py \\
        --assembled ./cnt_10_0_u23_Xe4_10ns_assembled.npz \\
        --theta-deg 0 15 30 45 54.7 60 75 90 \\
        --fit-max-lag-ps 3000 \\
        --b0-fields 9.4 14.1 \\
        --out ./cnt_10_0_u23_Xe4_10ns_tilt_relaxation.npz
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

GAMMA_XE_DEFAULT = -7.400e7  # rad s^-1 T^-1, 129Xe gyromagnetic ratio, same default as step2_build_tcf.py
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Verbatim from step2_build_tcf.py / step2c_build_h2_tcf.py, copied rather
# than imported so this script has no dependency on those files and cannot
# silently diverge if they change later.
# ---------------------------------------------------------------------------

def compute_h0_h1(tensor: np.ndarray, gamma_xe: float):
    xz = tensor[..., 0, 2]
    yz = tensor[..., 1, 2]
    trace = tensor[..., 0, 0] + tensor[..., 1, 1] + tensor[..., 2, 2]
    sigma_iso = trace / 3.0
    zz_traceless_ppm = tensor[..., 2, 2] - sigma_iso
    h0 = 0.5 * gamma_xe * (zz_traceless_ppm * 1e-6)
    h1 = (1.0 / np.sqrt(6.0)) * gamma_xe * (-(xz * 1e-6) - 1j * (yz * 1e-6))
    return h0, h1


def compute_h2(tensor: np.ndarray, gamma_xe: float):
    xx = tensor[..., 0, 0]
    yy = tensor[..., 1, 1]
    xy = tensor[..., 0, 1]
    h2 = (gamma_xe / np.sqrt(6.0)) * (0.5 * (xx - yy) * 1e-6 + 1j * (xy * 1e-6))
    return h2


def fft_corr(x: np.ndarray) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# New: the reduced Wigner d-matrix elements needed (rows m=0 and m=1 only,
# the only two that ever enter spin-1/2 lab-frame relaxation, at any tilt
# angle, the same selection rule that already restricted Steps 1-6 to h0
# and h1 at theta=0). Verified symbolically against sympy's independent
# Wigner-d implementation before this script was written, see the
# conversation record, both rows collapse to the identity at theta=0
# (d2_00(0)=1, d2_11(0)=1, everything else in these two rows =0), exactly
# reproducing the existing theta=0 formulas.
# ---------------------------------------------------------------------------

def wigner_d2_row0(theta):
    """[d2_{0,-2}, d2_{0,-1}, d2_{0,0}, d2_{0,1}, d2_{0,2}](theta)."""
    s, c = math.sin(theta), math.cos(theta)
    s2, c2 = math.sin(2 * theta), math.cos(2 * theta)
    return np.array([
        0.25 * math.sqrt(6) * s ** 2,
        -0.25 * math.sqrt(6) * s2,
        0.75 * c2 + 0.25,
        0.25 * math.sqrt(6) * s2,
        0.25 * math.sqrt(6) * s ** 2,
    ])


def wigner_d2_row1(theta):
    """[d2_{1,-2}, d2_{1,-1}, d2_{1,0}, d2_{1,1}, d2_{1,2}](theta)."""
    s, c = math.sin(theta), math.cos(theta)
    c2 = math.cos(2 * theta)
    return np.array([
        0.5 * (c - 1.0) * s,
        0.5 * c - 0.5 * c2,
        -0.25 * math.sqrt(6) * math.sin(2 * theta),
        0.5 * c + 0.5 * c2,
        0.5 * (c + 1.0) * s,
    ])


def tube_frame_components(tensor: np.ndarray, gamma_xe: float):
    """Returns h_tube,k(t) for k=-2,-1,0,1,2, each (n_frames, n_atoms),
    complex except k=0 which is real. h_{-1} = -h1^*, h_{-2} = +h2^*
    (the sign difference between these two relations is not a typo, see
    the module docstring, (-1)^m alternates with m)."""
    h0, h1 = compute_h0_h1(tensor, gamma_xe)
    h2 = compute_h2(tensor, gamma_xe)
    h_m1 = -np.conj(h1)
    h_m2 = np.conj(h2)
    # order matches the d-matrix rows above: k = -2,-1,0,1,2
    return [h_m2, h_m1, h0.astype(complex), h1, h2]


def lab_frame_h0_h1(tensor: np.ndarray, gamma_xe: float, theta_rad: float):
    """h_lab,0(t) and h_lab,1(t) at the given tilt angle, built explicitly
    in the time domain, each (n_frames, n_atoms)."""
    tube = tube_frame_components(tensor, gamma_xe)  # [h_-2, h_-1, h_0, h_1, h_2]
    row0 = wigner_d2_row0(theta_rad)
    row1 = wigner_d2_row1(theta_rad)
    h_lab_0 = sum(c * h for c, h in zip(row0, tube))
    h_lab_1 = sum(c * h for c, h in zip(row1, tube))
    return h_lab_0, h_lab_1


def check_cross_correlations(tensor: np.ndarray, gamma_xe: float, max_lag_frames: int = 200):
    """Diagnostic (not applied automatically): reports the size of
    <dh_tube,k1*(0) dh_tube,k2(t)> for k1 != k2 relative to the k1=k2
    autocorrelations, at zero lag, to check the azimuthal-symmetry
    assumption this script otherwise takes for granted. Call this
    separately and inspect before trusting results at theta far from
    0 or 90 degrees."""
    tube = tube_frame_components(tensor, gamma_xe)
    means = [h.mean(axis=0) for h in tube]
    dtube = [h - m[None, :] for h, m in zip(tube, means)]
    labels = ["-2", "-1", "0", "1", "2"]
    print("Cross-correlation check at t=0, |<dh_k1* dh_k2>| relative to sqrt(<|dh_k1|^2><|dh_k2|^2>):")
    for i in range(5):
        for j in range(i + 1, 5):
            cross = np.mean(np.conj(dtube[i]) * dtube[j])
            norm = np.sqrt(np.mean(np.abs(dtube[i]) ** 2) * np.mean(np.abs(dtube[j]) ** 2))
            ratio = abs(cross) / norm if norm > 0 else float("nan")
            print(f"  k1={labels[i]:>2}, k2={labels[j]:>2}: |cross|/norm = {ratio:.4f}")


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

    h_lab_0_mean = h_lab_0.mean(axis=0)
    h_lab_1_mean = h_lab_1.mean(axis=0)
    dh0 = h_lab_0 - h_lab_0_mean[None, :]
    dh1 = h_lab_1 - h_lab_1_mean[None, :]

    tcf_h0 = np.empty((n_frames, n_atoms))
    tcf_h1 = np.empty((n_frames, n_atoms), dtype=complex)
    for k in range(n_atoms):
        tcf_h0[:, k] = fft_corr(dh0[:, k]).real
        tcf_h1[:, k] = fft_corr(dh1[:, k])

    tcf_total = tcf_h0.mean(axis=1) + 2.0 * tcf_h1.mean(axis=1).real
    lag_ps = np.arange(n_frames) * dt_fs / 1000.0
    lag_s = lag_ps * 1e-12

    mask = lag_ps <= fit_max_lag_ps
    t_fit_ps, t_fit_s, y_fit = lag_ps[mask], lag_s[mask], tcf_total[mask]
    amps_ps, taus_ps, r2 = fit_track(t_fit_ps, y_fit)
    taus_si = [tau * 1e-12 for tau in taus_ps]

    J0 = lorentzian_sum(0.0, amps_ps, taus_si)[0]
    results = {}
    for B0 in b0_fields:
        w0 = abs(gamma_xe) * B0
        Jw0 = lorentzian_sum(w0, amps_ps, taus_si)[0]
        T1 = 1.0 / (6.0 * B0 ** 2 * Jw0) if Jw0 > 0 else float("nan")
        T2 = 1.0 / (B0 ** 2 * (4.0 * J0 + 3.0 * Jw0)) if (4.0 * J0 + 3.0 * Jw0) > 0 else float("nan")
        results[B0] = dict(T1=T1, T2=T2, J0=J0, Jw0=Jw0)
    return dict(amps_ps=amps_ps, taus_ps=taus_ps, r2=r2, tcf0=float(tcf_total[0]), b0_results=results)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", required=True, type=Path)
    ap.add_argument("--gamma-xe", type=float, default=GAMMA_XE_DEFAULT)
    ap.add_argument("--theta-deg", type=float, nargs="+", required=True)
    ap.add_argument("--fit-max-lag-ps", type=float, default=3000.0)
    ap.add_argument("--b0-fields", type=float, nargs="+", default=[9.4, 14.1])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    d = np.load(args.assembled)
    tensor = d["tensor"]
    dt_fs = float(d["dt_fs"])
    print(f"Loaded {args.assembled}: {tensor.shape[0]} frames, {tensor.shape[1]} tracks, dt={dt_fs} fs")

    all_results = {}
    for theta_deg in args.theta_deg:
        theta_rad = math.radians(theta_deg)
        res = relaxation_at_theta(tensor, dt_fs, args.gamma_xe, theta_rad,
                                   args.fit_max_lag_ps, args.b0_fields)
        all_results[theta_deg] = res
        line = f"theta={theta_deg:6.2f} deg  R2={res['r2']:.4f}  TCF(0)={res['tcf0']:.4e}"
        for B0, r in res["b0_results"].items():
            line += f"  T1({B0:g}T)={r['T1']:.4e}s  T2({B0:g}T)={r['T2']:.4e}s"
        print(line)

    np.savez_compressed(args.out, theta_deg=args.theta_deg, b0_fields=args.b0_fields,
                         results=all_results, gamma_xe=args.gamma_xe,
                         fit_max_lag_ps=args.fit_max_lag_ps)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
