#!/usr/bin/env python3
"""
Step 2c - Build h2(t) and its TCF, the third rank-2 spherical tensor
component of the shielding tensor, alongside the h0(t), h1(t) Step 2
already builds.

Why this exists: at theta=0 (SWCNT axis parallel to B0, the case this whole
project has computed so far), the m=+/-2 lab-frame component never enters
the spin-1/2 relaxation Hamiltonian at all (A^(+/-2)=0 is a Wigner-Eckart
selection rule for spin-1/2, independent of what is physically fluctuating).
At a general tilt angle between the SWCNT axis and B0, this is no longer
true, the lab-frame h0(t) and h1(t) become mixtures of ALL FIVE tube-frame
components once you rotate, including this one. Before building the full
tilt-angle formula, this script answers a narrower, cheaper question first,
is this component's spectral density even large enough to matter, using
data already sitting in the existing assembled.npz files, no new MD or
NMR-ML runs.

Convention, checked, not assumed: h0(t), h1(t) in step2_build_tcf.py are
exactly (gamma/sqrt6) times the standard normalized spherical tensor
components of the traceless shielding tensor (verified symbolically against
Man's Cartesian/spherical tensor formulas, both reproduce the code's actual
prefactors, 1/2 for m=0 and 1/sqrt6 for m=+1, from that single shared
factor). The consistent companion for m=+2 in that same convention is

    h2(t) = (gamma/sqrt6) * [ (1/2)(sigma_xx(t) - sigma_yy(t)) + i sigma_xy(t) ]

h2 is complex, same as h1. For a real (Hermitian), even-rank tensor,
h_{-2}(t) = +h2(t)* (note the sign: for h1 the analogous relation carries an
extra minus sign, h_{-1}(t) = -h1(t)*, since (-1)^m flips sign for odd m but
not even m), so only h2 (m=+2) needs to be computed and stored, exactly as
only h1 (not h_{-1}) is stored today.

The combined contribution this component would make to the total TCF, were
it required, is 2*Re<h2*(0)h2(t)>, mirroring exactly how h1's contribution
enters as 2*Re(tcf_h1_mean) in step2_build_tcf.py's tcf_total. This script
reports that quantity's zero-lag value alongside h0's and h1's own, so the
relative size is visible immediately without needing to load two files.

Usage
-----
    python step2c_build_h2_tcf.py \\
        --assembled ./cnt_10_0_u23_Xe4_10ns_assembled.npz \\
        --tcf ./cnt_10_0_u23_Xe4_10ns_tcf.npz \\
        --out ./cnt_10_0_u23_Xe4_10ns_h2_tcf.npz \\
        --plot ./cnt_10_0_u23_Xe4_10ns_h2_tcf.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

GAMMA_XE_DEFAULT = -7.400e7  # rad s^-1 T^-1, 129Xe gyromagnetic ratio, same default as step2_build_tcf.py


def compute_h2(tensor: np.ndarray, gamma_xe: float):
    """tensor: (n_frames, n_atoms, 3, 3) in ppm. Returns h2 (complex),
    (n_frames, n_atoms), in rad/s/T, same units as step2's h0, h1.

    h2(t) = (gamma/sqrt6) * [ (1/2)(sigma_xx(t) - sigma_yy(t)) + i sigma_xy(t) ]

    xx, yy enter only as a difference, so no trace/isotropic correction is
    needed (sigma_iso cancels identically), exactly as xz, yz needed none
    for h1. xy is likewise trace-independent.
    """
    xx = tensor[..., 0, 0]
    yy = tensor[..., 1, 1]
    xy = tensor[..., 0, 1]
    yx = tensor[..., 1, 0]
    asym = np.abs(xy - yx).max()
    if asym > 1e-3:
        print(f"WARNING: tensor is not as symmetric as expected (max |xy-yx| = {asym:.3e} ppm)")

    h2 = (gamma_xe / np.sqrt(6.0)) * (0.5 * (xx - yy) * 1e-6 + 1j * (xy * 1e-6))
    return h2


def fft_corr(x: np.ndarray) -> np.ndarray:
    """Identical to step2_build_tcf.py's fft_corr: linear (non-circular),
    lag-normalized correlation via zero-padded FFT. Kept as a verbatim copy
    rather than an import, so this script has no dependency on Step 2's
    file and cannot silently diverge if that file changes later."""
    n = x.shape[0]
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    X = np.fft.fft(x, n=nfft)
    S = np.conj(X) * X
    c = np.fft.ifft(S)[:n]
    counts = np.arange(n, 0, -1)
    return c / counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", required=True, type=Path, help="Step 1 output .npz")
    ap.add_argument("--tcf", required=True, type=Path,
                     help="Step 2 output .npz, read only for h0/h1 TCF(0) values, for the side-by-side comparison")
    ap.add_argument("--gamma-xe", type=float, default=GAMMA_XE_DEFAULT, help="129Xe gyromagnetic ratio, rad/s/T")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--plot", type=Path, default=None, help="optional PNG of the averaged h2 TCF, short-lag view")
    ap.add_argument("--plot-max-lag-ps", type=float, default=20.0)
    args = ap.parse_args()

    d = np.load(args.assembled)
    tensor = d["tensor"]
    dt_fs = float(d["dt_fs"])
    n_frames, n_atoms = tensor.shape[0], tensor.shape[1]
    print(f"Loaded {args.assembled}: {n_frames} frames, {n_atoms} tracks, dt={dt_fs} fs")

    h2 = compute_h2(tensor, args.gamma_xe)

    h2_mean_per_track = h2.mean(axis=0)  # (n_atoms,), complex
    print("Per-track mean h2 (static/residual biaxial contribution, rad/s/T, "
          "compare against h0's own static contribution, expected to also be nonzero under confinement):")
    for k, v in enumerate(h2_mean_per_track):
        print(f"    track {k}: {v:.4e}")

    dh2 = h2 - h2_mean_per_track[None, :]

    tcf_h2 = np.empty((n_frames, n_atoms), dtype=complex)
    for k in range(n_atoms):
        tcf_h2[:, k] = fft_corr(dh2[:, k])

    tcf_h2_mean = tcf_h2.mean(axis=1)
    lag_fs = np.arange(n_frames) * dt_fs
    lag_ps = lag_fs / 1000.0

    h2_contribution_0 = 2.0 * tcf_h2_mean[0].real
    print(f"\nh2 contribution to TCF(0), 2*Re<h2*(0)h2(0)> = {h2_contribution_0:.6e} rad^2 s^-2 T^-2")

    if args.tcf.exists():
        d_tcf = np.load(args.tcf)
        tcf_h0_0 = float(d_tcf["tcf_h0_mean"][0])
        h1_contribution_0 = 2.0 * float(d_tcf["tcf_h1_mean"][0].real)
        total_0 = float(d_tcf["tcf_total"][0])
        print(f"\nSide-by-side comparison at t=0 (rad^2 s^-2 T^-2):")
        print(f"  h0 contribution            : {tcf_h0_0:.6e}")
        print(f"  h1 contribution (2*Re part): {h1_contribution_0:.6e}")
        print(f"  h2 contribution (2*Re part): {h2_contribution_0:.6e}")
        print(f"  current tcf_total (h0+h1)  : {total_0:.6e}")
        if total_0 != 0:
            frac = h2_contribution_0 / total_0
            print(f"  h2 contribution / current tcf_total = {frac:.4f} "
                  f"({'small, plausibly negligible' if abs(frac) < 0.05 else 'NOT small, worth keeping in the tilt-angle formula'})")
    else:
        print(f"\nNote: {args.tcf} not found, skipping the side-by-side h0/h1 comparison, "
              f"only h2's own numbers are reported above.")

    np.savez_compressed(
        args.out,
        lag_fs=lag_fs,
        tcf_h2_mean=tcf_h2_mean,
        tcf_h2_per_track=tcf_h2,
        h2_mean_per_track=h2_mean_per_track,
        gamma_xe=args.gamma_xe,
        dt_fs=dt_fs,
    )
    print(f"\nWrote {args.out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mask = lag_ps <= args.plot_max_lag_ps
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(lag_ps[mask], 2.0 * tcf_h2_mean[mask].real, lw=1, label="h2 (2*Re part)")
        if args.tcf.exists():
            d_tcf = np.load(args.tcf)
            ax.plot(lag_ps[mask], d_tcf["tcf_total"][mask], lw=1, ls="--", label="current tcf_total (h0+h1)")
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("TCF contribution (rad$^2$ s$^{-2}$ T$^{-2}$)")
        ax.set_title("h2 contribution vs. current total, short-lag view")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
