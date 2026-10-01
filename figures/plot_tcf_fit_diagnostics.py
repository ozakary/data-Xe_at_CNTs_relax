#!/usr/bin/env python3
"""
Per-(system, theta) TCF-vs-fit diagnostic plot for the ESI, matching the
two-panel (data+fit+inset zoom on top, residual below) technique of
plot_time_correlation_20ps_vf_2.py, adapted to this project's bi/mono-
exponential TCF fits.

Unlike the old (theta=0-only) version of this script, which read the
already-saved fit from step3_spectral_density_and_relaxation.py's
_relaxation.npz, this one rebuilds the TCF from Step 1's assembled.npz
(the same Wigner-rotation combination used throughout the powder-averaged
methodology, copied verbatim, see plot_tcf_all_cases.py) and fits BOTH the
mono- and bi-exponential models explicitly at each requested angle.
step2f_powder_averaged_relaxation.py only keeps whichever one it actually
used per angle (bi if converged, else mono), not both separately, so this
diagnostic, whose whole purpose is showing both fits side by side, cannot
be built from its saved output alone.

Only 3 angles are plotted, not the full 10-point grid, to keep the ESI a
manageable size: theta = 0, 40, 90 degrees (parallel, an intermediate
angle, and perpendicular). Fit bounds (non-negative amplitudes) match
step2f_powder_averaged_relaxation.py's own corrected fit_pooled, the same
fix, applied here for the same reason.

For every system in the manifest and each of the 3 angles, writes one SVG
into its own results/<run_id>/ folder: {run_id}_fit_theta{00,40,90}.svg.

Usage
-----
    python plot_tcf_fit_diagnostics.py \\
        --manifest ./run_manifest.csv \\
        --results-dir ./results \\
        --theta-deg 0 40 90 \\
        --fit-max-lag-ps 3000 \\
        --xlim-max-ps 100 \\
        --inset-frac 0.2

Note: the fit itself always uses --fit-max-lag-ps (default 3000 ps, matching
print_tcf_fit_params_table.py and the rest of the pipeline), completely
independent of --xlim-max-ps, which only controls how much of the fitted
curve is DISPLAYED. An earlier version of this script incorrectly used
--xlim-max-ps for the fit itself, giving R^2 values that did not match
Tables S1-S6, fixed here.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit

try:
    import figure_formatting_v2 as ff
    ff.set_rcParams(ff.master_formatting)
    HAVE_FF = True
except ImportError:
    print("figure_formatting_v2 not found, using default matplotlib settings.")
    HAVE_FF = False

GAMMA_XE_DEFAULT = -7.451956e7  # rad s^-1 T^-1, 129Xe gyromagnetic ratio, matches step2_build_tcf.py


# ---------------------------------------------------------------------------
# Verbatim from plot_tcf_all_cases.py / step2f_powder_averaged_relaxation.py,
# copied rather than imported so this script has no dependency on those files.
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
    d = np.load(assembled_path)
    tensor = d["tensor"]
    dt_fs = float(d["dt_fs"])
    n_frames, n_atoms = tensor.shape[0], tensor.shape[1]
    h_lab_0, h_lab_1 = lab_frame_h0_h1(tensor, gamma_xe, theta_rad)
    dh0 = h_lab_0 - h_lab_0.mean(axis=0)[None, :]
    dh1 = h_lab_1 - h_lab_1.mean(axis=0)[None, :]
    tcf_h0 = np.empty((n_frames, n_atoms))
    tcf_h1 = np.empty((n_frames, n_atoms), dtype=complex)
    for k in range(n_atoms):
        tcf_h0[:, k] = fft_corr(dh0[:, k]).real
        tcf_h1[:, k] = fft_corr(dh1[:, k])
    tcf = tcf_h0.mean(axis=1) + 2.0 * tcf_h1.mean(axis=1).real
    lag_ps = np.arange(n_frames) * dt_fs / 1000.0
    return lag_ps, tcf


def mono_exp(t, A, tau):
    return A * np.exp(-t / tau)


def bi_exp(t, A1, tau1, A2, tau2):
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)


def fit_quality(t, y, model, popt):
    resid = y - model(t, *popt)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def fit_mono_and_bi(t_ps, y):
    """Fits BOTH models explicitly, unlike step2f's fit_pooled which only
    keeps whichever one it used. Same non-negative-amplitude bounds as
    step2f's corrected fit_pooled, same justification (see that file)."""
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
        return popt_mono, r2_mono, popt_bi, r2_bi
    except RuntimeError:
        return popt_mono, r2_mono, None, None


def plot_one_system_theta(run_id: str, results_dir: Path, theta_deg: float,
                           gamma_xe: float, fit_max_lag_ps: float, xlim_max_ps: float, inset_frac: float):
    sys_dir = results_dir / run_id
    assembled_path = sys_dir / f"{run_id}_assembled.npz"
    if not assembled_path.exists():
        print(f"Missing assembled.npz for {run_id}, skipping")
        return

    lag_ps, tcf = build_tcf_at_theta(assembled_path, gamma_xe, math.radians(theta_deg))

    # Fit over the FULL fit window (matches print_tcf_fit_params_table.py and
    # the rest of the pipeline), independent of how much gets displayed below.
    fit_mask = lag_ps <= fit_max_lag_ps
    t_fit, y_fit = lag_ps[fit_mask], tcf[fit_mask]
    mono_params, mono_r2, bi_params, bi_r2 = fit_mono_and_bi(t_fit, y_fit)
    bi_converged = bi_params is not None

    # Display only the requested x-range, evaluating the already-fit model
    # on this (generally shorter) range purely for plotting.
    mask = lag_ps <= xlim_max_ps
    t = lag_ps[mask]
    y = tcf[mask]

    mono_pred = mono_exp(t, *mono_params)

    def goodness(pred):
        return (1.0 - np.sum(np.abs(y - pred)) / max(np.sum(np.abs(y)), 1e-10)) * 100.0

    if bi_converged:
        bi_pred = bi_exp(t, *bi_params)
    else:
        bi_pred = None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9),
                                    gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # --- Top panel: TCF vs both fits ---
    ax1.plot(t, y, color="black", linewidth=2.5, label="TCF")
    ax1.plot(t, mono_pred, color="#3A86FF", linewidth=2.5, linestyle="--",
              label=f"Mono-exp ($R^2$={mono_r2:.3f})")
    if bi_converged:
        ax1.plot(t, bi_pred, color="#FF006E", linewidth=2.5, linestyle="-.",
                  label=f"Bi-exp ($R^2$={bi_r2:.3f})")
    else:
        ax1.text(0.02, 0.02, "bi-exponential fit did not converge", transform=ax1.transAxes,
                  fontsize=11, style="italic", color="0.4", ha="left", va="bottom")
    ax1.axhline(0, color="grey", linestyle="--", linewidth=2.5)
    ax1.set_ylabel(r"Shielding TCF / rad$^2$$\cdot$s$^{-2}$$\cdot$T$^{-2}$")
    ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
    leg = ax1.legend(frameon=False, loc="upper left")
    leg.get_frame().set_edgecolor("gray")
    leg.get_frame().set_alpha(0.25)

    all_curves = [y, mono_pred] + ([bi_pred] if bi_converged else [])
    y_min = min(c.min() for c in all_curves)
    y_max = max(c.max() for c in all_curves)
    margin = 0.05 * (y_max - y_min)
    ax1.set_ylim(y_min - margin, y_max + margin)
    ax1.set_xlim(0.0, xlim_max_ps)

    # --- Inset: zoom into the fast initial decay ---
    t_zoom_max = xlim_max_ps * inset_frac
    zoom_mask = t <= t_zoom_max
    axins = inset_axes(ax1, width="70%", height="70%", loc="lower right",
                        bbox_to_anchor=(0.28, 0.15, 0.70, 0.68), bbox_transform=ax1.transAxes)
    axins.plot(t[zoom_mask], y[zoom_mask], color="black", linewidth=2.5)
    axins.plot(t[zoom_mask], mono_pred[zoom_mask], color="#3A86FF", linewidth=2.5, linestyle="--")
    if bi_converged:
        axins.plot(t[zoom_mask], bi_pred[zoom_mask], color="#FF006E", linewidth=2.5, linestyle="-.")
    axins.axhline(0, color="grey", linestyle="--", linewidth=2.5)
    axins.set_xlim(0.0, 10.0)
    zoom_curves = [y[zoom_mask], mono_pred[zoom_mask]] + ([bi_pred[zoom_mask]] if bi_converged else [])
    yz_min = min(c.min() for c in zoom_curves)
    yz_max = max(c.max() for c in zoom_curves)
    zm = 0.05 * (yz_max - yz_min)
    axins.set_ylim(yz_min - zm, yz_max + zm)
    axins.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)

    # --- Bottom panel: residuals for both fits ---
    mono_resid = y - mono_pred
    ax2.plot(t, mono_resid, color="#3A86FF", linewidth=2.5, linestyle="--",
              label=f"Mono-exp ($G$={goodness(mono_pred):.1f}%)")
    resid_curves = [mono_resid]
    if bi_converged:
        bi_resid = y - bi_pred
        ax2.plot(t, bi_resid, color="#FF006E", linewidth=2.5, linestyle="-.",
                  label=f"Bi-exp ($G$={goodness(bi_pred):.1f}%)")
        resid_curves.append(bi_resid)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=2.5)
    ax2.set_xlabel(r"$t$ / ps")
    ax2.set_ylabel(r"$\Delta$TCF")
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
    ax2.legend(frameon=False, fontsize=14, loc="upper right")
    r_abs = max(max(np.abs(c).max() for c in resid_curves), 1e-10)
    ax2.set_ylim(-r_abs * 1.2, r_abs * 1.2)
    ax2.set_xlim(0.0, xlim_max_ps)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.15)

    out_path = sys_dir / f"{run_id}_fit_theta{theta_deg:02.0f}.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--theta-deg", type=float, nargs="+", default=[0, 40, 90],
                     help="only these angles are plotted, kept few to keep the ESI a manageable size")
    ap.add_argument("--gamma-xe", type=float, default=GAMMA_XE_DEFAULT)
    ap.add_argument("--fit-max-lag-ps", type=float, default=3000.0,
                     help="the fit itself uses [0, this] (ps), independent of the display range below, "
                          "must match print_tcf_fit_params_table.py's --fit-max-lag-ps for the R^2 values "
                          "in the tables and these figures to agree")
    ap.add_argument("--xlim-max-ps", type=float, default=100.0,
                     help="main panel and residual panel DISPLAY [0, this] (ps), does not affect the fit itself")
    ap.add_argument("--inset-frac", type=float, default=0.2,
                     help="inset shows [0, this fraction of --xlim-max-ps]")
    ap.add_argument("--run-id", type=str, default=None,
                     help="only plot this one system, default: every system in the manifest")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    run_ids = [args.run_id] if args.run_id else manifest["run_id"].tolist()
    for run_id in run_ids:
        for theta_deg in args.theta_deg:
            plot_one_system_theta(run_id, args.results_dir, theta_deg, args.gamma_xe,
                                   args.fit_max_lag_ps, args.xlim_max_ps, args.inset_frac)


if __name__ == "__main__":
    main()
