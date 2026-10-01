#!/usr/bin/env python3
"""
Step 2 - Build the CSA perturbation-Hamiltonian components h0(t), h1(t) from
the tracked shielding tensor (Step 1 output), and compute the shielding time
correlation function (TCF), following Hanni, Lantto, Vaara,
Phys. Chem. Chem. Phys. 2011, 13, 13704, eqs. (9)-(11) of the SI.

    h0(t) = (1/2)  gamma_Xe [sigma_ZZ(t) - sigma_iso(t)]
    h1(t) = (1/sqrt6) gamma_Xe [-sigma_XZ(t) - i sigma_YZ(t)]
    TCF(t) = <dh0(0)dh0(t)> + 2 Re<dh1(0)* dh1(t)>

where dh = h - <h> is the deviation from the per-track time average. Only
fluctuations drive Redfield relaxation, the mean part of h is a coherent,
non-relaxing contribution (a static/residual CSA), not a relaxation source.
This matters here specifically because the CNT axis is a fixed lab direction,
unlike the isotropically-sampled free gas in the reference paper, so <h0> is
not guaranteed to vanish by symmetry the way it does for gas-phase Xe, and in
practice for this system it does not: the raw (non-mean-subtracted) TCF's
running integral was found to grow roughly linearly out to the full 10 ns
trajectory instead of converging, the signature of a non-zero long-time
mean rather than a real, very slow decay.

The per-track mean h0, h1 are reported and saved separately, they are a
physically meaningful "static confinement anisotropy" in their own right,
just not part of the relaxation calculation.

sigma_ZZ, sigma_XZ, sigma_YZ, sigma_iso are the lab-frame (box-frame, CNT
along z) components of the shielding tensor from Step 1's assembled array,
in ppm, converted to dimensionless (x1e-6) before use.

The ACF (for h0) and cross-correlation (for h1) are computed per Xe track
via zero-padded FFT (Wiener-Khinchin), then averaged over tracks.

Usage
-----
    python step2_build_tcf.py \\
        --assembled ./cnt_10_0_u23_Xe4_10ns_assembled.npz \\
        --out ./cnt_10_0_u23_Xe4_10ns_tcf.npz \\
        --plot ./cnt_10_0_u23_Xe4_10ns_tcf.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

GAMMA_XE_DEFAULT = -7.451956e7  # rad s^-1 T^-1, 129Xe gyromagnetic ratio.


def compute_h0_h1(tensor: np.ndarray, gamma_xe: float):
    """tensor: (n_frames, n_atoms, 3, 3) in ppm. Returns h0 (real) and h1
    (complex), each (n_frames, n_atoms), in rad/s/T."""
    xz = tensor[..., 0, 2]
    zx = tensor[..., 2, 0]
    yz = tensor[..., 1, 2]
    zy = tensor[..., 2, 1]
    asym = np.maximum(np.abs(xz - zx).max(), np.abs(yz - zy).max())
    if asym > 1e-3:
        print(f"WARNING: tensor is not as symmetric as expected (max |off-diag mismatch| = {asym:.3e} ppm)")

    trace = tensor[..., 0, 0] + tensor[..., 1, 1] + tensor[..., 2, 2]
    sigma_iso = trace / 3.0
    zz_traceless_ppm = tensor[..., 2, 2] - sigma_iso

    h0 = 0.5 * gamma_xe * (zz_traceless_ppm * 1e-6)
    h1 = (1.0 / np.sqrt(6.0)) * gamma_xe * (-(xz * 1e-6) - 1j * (yz * 1e-6))
    return h0, h1


def fft_corr(x: np.ndarray) -> np.ndarray:
    """Linear (non-circular), lag-normalized correlation C[lag] =
    (1/(N-lag)) sum_t conj(x[t]) x[t+lag], for lag = 0..N-1, via zero-padded
    FFT. Works for real or complex x. Pass in the fluctuation (mean already
    removed), not the raw signal."""
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
    ap.add_argument("--gamma-xe", type=float, default=GAMMA_XE_DEFAULT, help="129Xe gyromagnetic ratio, rad/s/T")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--plot", type=Path, default=None, help="optional PNG of the averaged TCF, short-lag view")
    ap.add_argument("--plot-max-lag-ps", type=float, default=20.0)
    args = ap.parse_args()

    d = np.load(args.assembled)
    tensor = d["tensor"]
    dt_fs = float(d["dt_fs"])
    n_frames, n_atoms = tensor.shape[0], tensor.shape[1]
    print(f"Loaded {args.assembled}: {n_frames} frames, {n_atoms} tracks, dt={dt_fs} fs")

    h0, h1 = compute_h0_h1(tensor, args.gamma_xe)

    h0_mean_per_track = h0.mean(axis=0)          # (n_atoms,)
    h1_mean_per_track = h1.mean(axis=0)          # (n_atoms,), complex
    print("Per-track mean h0 (static/residual axial CSA contribution, rad/s/T):")
    for k, v in enumerate(h0_mean_per_track):
        print(f"    track {k}: {v:.4e}")
    print("Per-track mean h1 (should be small/zero if the tube has enough azimuthal symmetry sampled):")
    for k, v in enumerate(h1_mean_per_track):
        print(f"    track {k}: {v:.4e}")

    dh0 = h0 - h0_mean_per_track[None, :]
    dh1 = h1 - h1_mean_per_track[None, :]

    tcf_h0 = np.empty((n_frames, n_atoms))
    tcf_h1 = np.empty((n_frames, n_atoms), dtype=complex)
    for k in range(n_atoms):
        tcf_h0[:, k] = fft_corr(dh0[:, k]).real
        tcf_h1[:, k] = fft_corr(dh1[:, k])

    tcf_h0_mean = tcf_h0.mean(axis=1)
    tcf_h1_mean = tcf_h1.mean(axis=1)
    tcf_total = tcf_h0_mean + 2.0 * tcf_h1_mean.real

    lag_fs = np.arange(n_frames) * dt_fs
    lag_ps = lag_fs / 1000.0

    print(f"\nTCF(0) = {tcf_total[0]:.6e} rad^2 s^-2 T^-2  "
          f"(h0 part {tcf_h0_mean[0]:.3e}, 2*Re(h1 part) {2*tcf_h1_mean[0].real:.3e})")
    tail = tcf_total[-len(tcf_total)//10:]
    print(f"Mean of the last 10% of lags (noise floor check): {tail.mean():.3e}, std {tail.std():.3e}")

    np.savez_compressed(
        args.out,
        lag_fs=lag_fs,
        tcf_total=tcf_total,
        tcf_h0_mean=tcf_h0_mean,
        tcf_h1_mean=tcf_h1_mean,
        tcf_h0_per_track=tcf_h0,
        tcf_h1_per_track=tcf_h1,
        h0_mean_per_track=h0_mean_per_track,
        h1_mean_per_track=h1_mean_per_track,
        gamma_xe=args.gamma_xe,
        dt_fs=dt_fs,
    )
    print(f"Wrote {args.out}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mask = lag_ps <= args.plot_max_lag_ps
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(lag_ps[mask], tcf_total[mask], lw=1)
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("shielding TCF (rad$^2$ s$^{-2}$ T$^{-2}$)")
        ax.set_title("Total shielding TCF (fluctuation-based), short-lag view")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
