#!/usr/bin/env python3
"""
Step 4 - Per-track uncertainty on T1, T2.

Step 3 fits the 4-track-averaged TCF and gives one point estimate. Since
Step 2 already saved each track's own TCF (tcf_h0_per_track, tcf_h1_per_track),
we can fit each of the 4 Xe atoms independently and get a real spread, rather
than a single number with no uncertainty. This mirrors the per-track spread
you'd actually report in the letter, rather than an error margin borrowed
from somewhere else.

Fit bounds (added after finding a case where an unbounded per-track fit
converged to a spurious long-tau, near-zero-amplitude component): a single
atom's own TCF is much noisier than the 4-track average Step 3 fits, and an
unconstrained bi-exponential fit can find a component with tiny amplitude
and enormous tau that barely affects the time-domain fit (so R^2 still
looks fine) but corrupts J(w0), since J(0) = A*tau can stay sizeable while
J(w0) collapses toward zero as (w0*tau)^2 blows up, sending that one
track's T1 to an unphysical value that then dominates the mean and SEM
across only 4 tracks. tau is now capped at the fit window itself, since a
correlation time longer than the data you're fitting over isn't something
a single noisy track can actually determine, and amplitudes are capped at
a generous multiple of the data's own scale.

Usage
-----
    python step4_per_track_uncertainty.py \\
        --tcf ./cnt_10_0_u23_Xe4_10ns_tcf.npz \\
        --fit-max-lag-ps 3000 \\
        --b0-fields 9.4 14.1 \\
        --out ./cnt_10_0_u23_Xe4_10ns_relaxation_per_track.npz
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


def fit_track(t_ps, y, fit_max_lag_ps):
    # tau capped at the fit window: a correlation time this data can't span
    # isn't something a single noisy track can reliably determine anyway.
    # Amplitudes constrained to be non-negative, matching Hanni et al.'s own
    # bi-exponential form (C*exp(-t/tau_p) + D*exp(-t/tau_t), both
    # coefficients implicitly positive): with tau > 0 and every amplitude
    # >= 0, J(omega) = sum_i A_i*tau_i/(1+(omega*tau_i)^2) is a sum of
    # strictly non-negative terms for any omega, so T1 and T2 cannot come
    # out negative or blow up from a spurious near-zero J(w0) afterward,
    # this is a guarantee, not just a reduced chance, and it's exactly the
    # failure mode found in a real per-track fit (one track's T2 came out
    # at 1.3e6 s, two others came out negative, all from amplitude sign and
    # tau magnitude being left completely unconstrained).
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
            bounds=([0.0, 1e-3, 0.0, 1e-3],
                    [amp_bound, fit_max_lag_ps, amp_bound, fit_max_lag_ps]),
        )
        resid = y - bi_exp(t_ps, *popt_bi)
        r2 = 1.0 - np.sum(resid**2) / np.sum((y - y.mean())**2)
        return [popt_bi[0], popt_bi[2]], [popt_bi[1], popt_bi[3]], r2
    except RuntimeError:
        resid = y - mono_exp(t_ps, *popt_mono)
        r2 = 1.0 - np.sum(resid**2) / np.sum((y - y.mean())**2)
        return [popt_mono[0]], [popt_mono[1]], r2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tcf", required=True, type=Path)
    ap.add_argument("--fit-max-lag-ps", type=float, default=1000.0)
    ap.add_argument("--b0-fields", type=float, nargs="+", default=[9.4, 14.1])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    d = np.load(args.tcf)
    lag_fs = d["lag_fs"]
    gamma_xe = float(d["gamma_xe"])
    tcf_h0_pt = d["tcf_h0_per_track"]     # (n_frames, n_atoms)
    tcf_h1_pt = d["tcf_h1_per_track"]     # (n_frames, n_atoms), complex
    n_frames, n_atoms = tcf_h0_pt.shape
    lag_ps = lag_fs / 1000.0
    mask = lag_ps <= args.fit_max_lag_ps
    t_fit_ps = lag_ps[mask]
    print(f"Fitting {n_atoms} tracks independently, {mask.sum()} points out to {t_fit_ps[-1]:.1f} ps each")

    per_track_T1 = {B0: [] for B0 in args.b0_fields}
    per_track_T2 = {B0: [] for B0 in args.b0_fields}
    per_track_r2 = []

    for k in range(n_atoms):
        y = (tcf_h0_pt[:, k] + 2.0 * tcf_h1_pt[:, k].real)[mask]
        amps_ps, taus_ps, r2 = fit_track(t_fit_ps, y, args.fit_max_lag_ps)
        per_track_r2.append(r2)
        amps_si = amps_ps
        taus_si = [tau * 1e-12 for tau in taus_ps]
        J0 = lorentzian_sum(0.0, amps_si, taus_si)[0]
        track_T1s, track_T2s = [], []
        for B0 in args.b0_fields:
            w0 = abs(gamma_xe) * B0
            Jw0 = lorentzian_sum(w0, amps_si, taus_si)[0]
            T1 = 1.0 / (6.0 * B0**2 * Jw0)
            T2 = 1.0 / (B0**2 * (4.0 * J0 + 3.0 * Jw0))
            per_track_T1[B0].append(T1)
            per_track_T2[B0].append(T2)
            track_T1s.append(T1)
            track_T2s.append(T2)
        print(f"  track {k}: R^2={r2:.4f}, taus_ps={[round(t,3) for t in taus_ps]}, "
              f"T1={[round(t,1) for t in track_T1s]}, T2={[round(t,1) for t in track_T2s]}")

    print()
    print(f"{'B0 (T)':>8} {'T1 mean (s)':>14} {'T1 std (s)':>12} {'T1 SEM (s)':>12} "
          f"{'T2 mean (s)':>14} {'T2 std (s)':>12} {'T2 SEM (s)':>12}")
    summary = {}
    for B0 in args.b0_fields:
        t1 = np.array(per_track_T1[B0])
        t2 = np.array(per_track_T2[B0])
        sem1 = t1.std(ddof=1) / np.sqrt(n_atoms)
        sem2 = t2.std(ddof=1) / np.sqrt(n_atoms)
        print(f"{B0:8.2f} {t1.mean():14.4e} {t1.std(ddof=1):12.4e} {sem1:12.4e} "
              f"{t2.mean():14.4e} {t2.std(ddof=1):12.4e} {sem2:12.4e}")

        # Flag, but do not silently drop, any track whose T1 or T2 is far
        # from the others: a real, wide per-track spread is possible, but a
        # lone outlier this large is worth a manual look before trusting
        # the mean. Also assert positivity directly: with amplitudes and
        # tau now bounded non-negative, J(w) >= 0 everywhere, so T1 and T2
        # cannot come out <= 0 mathematically, if one ever does, that is a
        # sign something else (not the fit-degeneracy issue this bound was
        # written for) is wrong, and needs a fresh look, not a dropped point.
        for label, arr in (("T1", t1), ("T2", t2)):
            assert np.all(arr > 0), (
                f"{label} at {B0:g} T came out <= 0 for at least one track after bounding "
                f"amplitudes/tau to be physically sensible, this should be impossible, "
                f"investigate before trusting anything from this system: {arr}"
            )
        med1, med2 = np.median(t1), np.median(t2)
        out1 = np.where(np.abs(t1 - med1) > 5 * med1)[0]
        out2 = np.where(np.abs(t2 - med2) > 5 * med2)[0]
        if len(out1) and len(out1) < n_atoms:
            print(f"  WARNING at {B0:g} T: track(s) {list(out1)} have T1 more than 5x the "
                  f"median of the others ({med1:.1f} s), inspect before trusting mean/SEM.")
        if len(out2) and len(out2) < n_atoms:
            print(f"  WARNING at {B0:g} T: track(s) {list(out2)} have T2 more than 5x the "
                  f"median of the others ({med2:.1f} s), inspect before trusting mean/SEM.")

        summary[B0] = dict(T1_mean=t1.mean(), T1_std=t1.std(ddof=1), T1_sem=sem1,
                            T2_mean=t2.mean(), T2_std=t2.std(ddof=1), T2_sem=sem2,
                            T1_per_track=t1, T2_per_track=t2)

    print(f"\nPer-track bi-exponential fit quality (R^2): {[round(r,3) for r in per_track_r2]}")
    if min(per_track_r2) < 0.7:
        print("NOTE: at least one track fits noticeably worse than the others, worth a look at that "
              "track's own TCF before trusting the pooled uncertainty.")

    np.savez_compressed(args.out, b0_fields=args.b0_fields, summary=summary, per_track_r2=per_track_r2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
