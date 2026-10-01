#!/usr/bin/env python3
"""
Step 3 - Spectral density and T1/T2, from the Step 2 TCF.

Rationale for the approach (matches Hanni, Lantto, Vaara 2011, comparing
"plateau/exponential fit" against the directly simulated SDF):

  The raw TCF is trustworthy out to roughly where its running integral
  stops climbing (visible in Step 2b's diagnostic plot), beyond that, a
  single 10 ns trajectory just doesn't have enough independent time-origins
  left to estimate it reliably, and integrating the noisy tail directly
  would corrupt J(0). So:

    1. Fit TCF(t) for t <= --fit-max-lag-ps to a sum of 1 or 2 exponentials.
       This has an exact analytic Fourier transform, J_fit(w) =
       sum_i A_i tau_i / (1 + (w tau_i)^2), so it can be evaluated at any
       omega, including the Larmor frequency, without ever touching the
       noisy tail.
    2. As a cross-check, also numerically integrate the raw (unfitted) TCF,
       trapezoidal quadrature, cos(w t), up to the same cutoff. Direct and
       fit-based should agree reasonably if the fit is a fair description
       of the trustworthy region.
    3. Evaluate J at omega=0 and at the Larmor frequency for each requested
       B0 field, then get T1, T2 from
           1/T1 = 6 B0^2 J(w0)
           1/T2 = B0^2 [4 J(0) + 3 J(w0)]
       (Hanni et al., main text eq. 1).

Usage
-----
    python step3_spectral_density_and_relaxation.py \\
        --tcf ./cnt_10_0_u23_Xe4_10ns_tcf.npz \\
        --fit-max-lag-ps 1000 \\
        --b0-fields 9.4 14.1 \\
        --out ./cnt_10_0_u23_Xe4_10ns_relaxation.npz \\
        --plot ./cnt_10_0_u23_Xe4_10ns_fit.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

_trapz = getattr(np, "trapezoid", None) or np.trapz


def mono_exp(t, A, tau):
    return A * np.exp(-t / tau)


def bi_exp(t, A1, tau1, A2, tau2):
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)


def lorentzian_sum(omega, amps, taus):
    """J(omega) for a sum of exponentials with the given amplitudes/taus."""
    omega = np.atleast_1d(omega).astype(float)
    J = np.zeros_like(omega)
    for A, tau in zip(amps, taus):
        J += A * tau / (1.0 + (omega * tau) ** 2)
    return J


def fit_quality(t, y, model, popt):
    resid = y - model(t, *popt)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot  # R^2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tcf", required=True, type=Path)
    ap.add_argument("--fit-max-lag-ps", type=float, default=1000.0,
                     help="only fit/integrate the TCF out to this lag, beyond this a single 10 ns "
                          "trajectory is typically too noisy to trust (check Step 2b's running-integral plot)")
    ap.add_argument("--b0-fields", type=float, nargs="+", default=[9.4, 14.1], help="Tesla")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.tcf)
    lag_fs = d["lag_fs"]
    tcf = d["tcf_total"]
    gamma_xe = float(d["gamma_xe"])
    lag_ps = lag_fs / 1000.0
    lag_s = lag_fs * 1e-15

    mask = lag_ps <= args.fit_max_lag_ps
    t_fit_ps = lag_ps[mask]
    t_fit_s = lag_s[mask]
    y_fit = tcf[mask]
    print(f"Using {mask.sum()} points out to {t_fit_ps[-1]:.1f} ps for fitting/direct integration")

    # --- mono-exponential fit ---
    p0_mono = [y_fit[0], 2.0]
    popt_mono, _ = curve_fit(mono_exp, t_fit_ps, y_fit, p0=p0_mono, maxfev=20000)
    r2_mono = fit_quality(t_fit_ps, y_fit, mono_exp, popt_mono)
    print(f"Mono-exponential fit: A={popt_mono[0]:.4e}, tau={popt_mono[1]:.3f} ps, R^2={r2_mono:.4f}")

    # --- bi-exponential fit ---
    p0_bi = [y_fit[0] * 0.8, 1.0, y_fit[0] * 0.2, 50.0]
    try:
        popt_bi, _ = curve_fit(bi_exp, t_fit_ps, y_fit, p0=p0_bi, maxfev=50000,
                                bounds=([-np.inf, 1e-3, -np.inf, 1e-3], [np.inf, np.inf, np.inf, np.inf]))
        r2_bi = fit_quality(t_fit_ps, y_fit, bi_exp, popt_bi)
        print(f"Bi-exponential fit: A1={popt_bi[0]:.4e}, tau1={popt_bi[1]:.3f} ps, "
              f"A2={popt_bi[2]:.4e}, tau2={popt_bi[3]:.3f} ps, R^2={r2_bi:.4f}")
        amps_ps = [popt_bi[0], popt_bi[2]]
        taus_ps = [popt_bi[1], popt_bi[3]]
    except RuntimeError as e:
        print(f"Bi-exponential fit did not converge ({e}), falling back to mono-exponential for J(w)")
        amps_ps = [popt_mono[0]]
        taus_ps = [popt_mono[1]]

    # amplitudes/taus in SI units (s) for J(w) in SI (rad^2 s^-1 T^-2), so 1/T1, 1/T2 come out in s^-1
    amps_si = amps_ps
    taus_si = [tau * 1e-12 for tau in taus_ps]

    def J_fit(omega):
        return lorentzian_sum(omega, amps_si, taus_si)

    def J_direct(omega):
        """Direct trapezoidal quadrature of the raw (unfitted) TCF x cos(w t), truncated at the fit cutoff."""
        return _trapz(y_fit * np.cos(omega * t_fit_s), t_fit_s)

    print()
    print(f"{'B0 (T)':>8} {'w0 (rad/s)':>14} {'J(0) fit':>12} {'J(0) direct':>12} "
          f"{'J(w0) fit':>12} {'J(w0) direct':>12} {'T1 (s)':>12} {'T2 (s)':>12}")

    results = []
    J0_fit = J_fit(0.0)[0]
    J0_direct = J_direct(0.0)
    for B0 in args.b0_fields:
        w0 = abs(gamma_xe) * B0
        Jw0_fit = J_fit(w0)[0]
        Jw0_direct = J_direct(w0)
        T1 = 1.0 / (6.0 * B0 ** 2 * Jw0_fit)
        T2 = 1.0 / (B0 ** 2 * (4.0 * J0_fit + 3.0 * Jw0_fit))
        print(f"{B0:8.2f} {w0:14.4e} {J0_fit:12.4e} {J0_direct:12.4e} "
              f"{Jw0_fit:12.4e} {Jw0_direct:12.4e} {T1:12.4e} {T2:12.4e}")
        results.append(dict(B0=B0, w0=w0, J0_fit=J0_fit, J0_direct=J0_direct,
                             Jw0_fit=Jw0_fit, Jw0_direct=Jw0_direct, T1=T1, T2=T2))

    np.savez_compressed(
        args.out,
        fit_max_lag_ps=args.fit_max_lag_ps,
        mono_fit_params=popt_mono,
        mono_fit_r2=r2_mono,
        bi_or_mono_amps=amps_ps,
        bi_or_mono_taus=taus_ps,
        b0_fields=args.b0_fields,
        results=results,
    )
    print(f"\nWrote {args.out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        ax = axes[0]
        ax.plot(t_fit_ps, y_fit, lw=1, label="TCF (fit region)")
        ax.plot(t_fit_ps, mono_exp(t_fit_ps, *popt_mono), "--", lw=1, label=f"mono exp (R2={r2_mono:.3f})")
        if len(amps_ps) == 2:
            ax.plot(t_fit_ps, bi_exp(t_fit_ps, *popt_bi), ":", lw=1.5, label=f"bi exp (R2={r2_bi:.3f})")
        ax.set_xlabel("time (ps)"); ax.set_ylabel("TCF"); ax.legend(); ax.set_title("Fit region")

        ax = axes[1]
        w_plot = np.logspace(6, 13, 400)
        ax.loglog(w_plot, J_fit(w_plot), label="J(w), fit-based")
        for B0 in args.b0_fields:
            w0 = abs(gamma_xe) * B0
            ax.axvline(w0, color="gray", lw=1, ls=":")
            ax.text(w0, ax.get_ylim()[1], f"{B0}T", rotation=90, fontsize=8, va="top")
        ax.set_xlabel("omega (rad/s)"); ax.set_ylabel("J(omega)"); ax.set_title("Spectral density")

        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
