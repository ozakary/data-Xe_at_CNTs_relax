#!/usr/bin/env python3
"""
TCF vs time, one SVG per (case, theta), all systems of each case overlaid,
following code_plot_rdf_xe-c_1d.py's coloring convention (RdBu_r,
cool-to-hot split by the swept variable) and figure_formatting_v2.py's
fixed-frame figure system. Series (LL/HL, or (10,0)/(30,0)) are
distinguished by linestyle, solid for the first, dashed for the second.
Axis labels follow Hanni, Lantto, Vaara 2011's wording ("shielding
autocorrelation function"), in our own actual units.

This is the theta-resolved companion to the old (theta=0-only)
plot_tcf_all_cases.py. Unlike plot_spectral_density_all_cases.py's
theta-resolved companion, this one cannot just read an already-saved
value: the raw per-angle TCF time series was never persisted by
step2f_powder_averaged_relaxation.py (only the fit parameters were, the
raw correlation function is built, fit, and discarded internally). So this
script rebuilds it from Step 1's assembled.npz, the same Wigner-rotation
combination (tube-frame h0/h1/h2 -> lab-frame h0/h1 -> FFT autocorrelation)
that step2f/step2g/step6(new) already use, copied verbatim here for the
same reason as those, no cross-file dependency. For a given case, every
system's TCF is built once, at all 10 angles, then reused across all 10
output SVGs for that case, rather than rebuilding per (case, theta) pair.

Uses a broken x-axis (two panels, left = short-time zoom, right = longer
tail), same technique as plot_time_correlation_20ps_vf_2.py: a gridspec
with two subplots sharing y, inner spines hidden at the break, and
hand-drawn diagonal break marks positioned from the actual axes boxes
after a draw pass. This exists because the interesting decay happens
within the first few ps, and a single linear time axis compresses it
into an unreadably thin sliver against a long, mostly-flat tail.

Writes 30 SVGs directly into --results-dir: case{1,2,3}_tcf_theta{00,10,
...,90}.svg. theta=00 is the one intended for the main text (matching the
pre-existing single-angle figure this project started with), the other
nine are ESI-only.

Usage
-----
    python plot_tcf_all_cases.py \\
        --manifest ./run_manifest.csv \\
        --results-dir ./results \\
        --x-left 0 5 \\
        --x-right 5 100
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
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

try:
    import figure_formatting_v2 as ff
    ff.set_rcParams(ff.master_formatting)
    HAVE_FF = True
except ImportError:
    print("figure_formatting_v2 not found, using default matplotlib settings.")
    HAVE_FF = False

SERIES_COLORS = ["#264653", "#2a9d8f"]
SERIES_LINESTYLES = ["-", "--"]

GAMMA_XE_DEFAULT = -7.451956e7  # rad s^-1 T^-1, 129Xe gyromagnetic ratio, matches step2_build_tcf.py

# Panel width ratios, NOT proportional to the time span each panel covers.
# The fast component (tau1 ~ 0.3 ps) decays almost instantly, so the left
# panel doesn't need much width to show it clearly. The slower component
# and the approach to the noise floor happen over the much longer right
# panel, which gets most of the horizontal space.
RATIO_LEFT = 0.3
RATIO_RIGHT = 0.7

CASE_CONFIG = {
    1: dict(x_label="$T$ / K", subcase_order=["low", "high"],
            subcase_display={"low": "LL", "high": "HL"}),
    2: dict(x_label=r"$d$ / $\mathrm{\AA}$", subcase_order=["low", "high"],
            subcase_display={"low": "LL", "high": "HL"}),
    3: dict(x_label="Xe loading / % max", subcase_order=["cnt_10_0", "cnt_30_0"],
            subcase_display={"cnt_10_0": "(10,0)", "cnt_30_0": "(30,0)"}),
}


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


def build_tcf_all_angles(assembled_path: Path, gamma_xe: float, theta_grid_deg):
    """Returns (lag_ps, {theta_deg: tcf_array}) for one system, building
    each angle's pooled TCF once."""
    d = np.load(assembled_path)
    tensor = d["tensor"]
    dt_fs = float(d["dt_fs"])
    n_frames, n_atoms = tensor.shape[0], tensor.shape[1]
    lag_ps = np.arange(n_frames) * dt_fs / 1000.0

    tcf_by_theta = {}
    for theta_deg in theta_grid_deg:
        h_lab_0, h_lab_1 = lab_frame_h0_h1(tensor, gamma_xe, math.radians(theta_deg))
        dh0 = h_lab_0 - h_lab_0.mean(axis=0)[None, :]
        dh1 = h_lab_1 - h_lab_1.mean(axis=0)[None, :]
        tcf_h0 = np.empty((n_frames, n_atoms))
        tcf_h1 = np.empty((n_frames, n_atoms), dtype=complex)
        for k in range(n_atoms):
            tcf_h0[:, k] = fft_corr(dh0[:, k]).real
            tcf_h1[:, k] = fft_corr(dh1[:, k])
        tcf_by_theta[theta_deg] = tcf_h0.mean(axis=1) + 2.0 * tcf_h1.mean(axis=1).real
    return lag_ps, tcf_by_theta


def variable_colors(values):
    """RdBu_r, cool half (dark blue -> mid blue) for the low end of the
    swept variable, hot half (mid red -> dark red) for the high end.
    Same convention as code_plot_rdf_xe-c_1d.py's color_positions."""
    unique_vals = sorted(set(values))
    n = len(unique_vals)
    try:
        cmap = matplotlib.colormaps["RdBu_r"]
    except AttributeError:
        cmap = plt.cm.get_cmap("RdBu_r")
    n_lower, n_upper = (n + 1) // 2, n // 2
    color_positions = np.concatenate([
        np.linspace(0.05, 0.25, n_lower),
        np.linspace(0.75, 0.95, n_upper),
    ])
    val_to_color = {v: cmap(color_positions[i]) for i, v in enumerate(unique_vals)}
    return val_to_color, cmap


def plot_case_tcf_theta(cdf: pd.DataFrame, config, theta_deg: float, tcf_cache: dict,
                         val_to_color, cmap, vmin, vmax, x_left, x_right, out_path: Path):
    fig = plt.figure(figsize=(6.0, 5.0))
    gap = 0.08
    gs = fig.add_gridspec(
        1, 2, left=0.16, right=0.75, bottom=0.18, top=0.90,
        width_ratios=[RATIO_LEFT, RATIO_RIGHT],
        wspace=gap * 2 / (RATIO_LEFT + RATIO_RIGHT),
    )
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1], sharey=ax_left)

    for ax in (ax_left, ax_right):
        ax.axhline(0, color="gray", linewidth=2, zorder=0)

    n_plotted = 0
    for _, row in cdf.iterrows():
        run_id = row["run_id"]
        if run_id not in tcf_cache:
            continue
        lag_ps, tcf_by_theta = tcf_cache[run_id]
        if theta_deg not in tcf_by_theta:
            continue
        tcf = tcf_by_theta[theta_deg]
        color = val_to_color[row["variable_value"]]
        ls = SERIES_LINESTYLES[config["subcase_order"].index(row["subcase"])]
        ax_left.plot(lag_ps, tcf, color=color, linestyle=ls, linewidth=2)
        ax_right.plot(lag_ps, tcf, color=color, linestyle=ls, linewidth=2)
        n_plotted += 1

    ax_left.set_xlim(*x_left)
    ax_right.set_xlim(*x_right)
    ax_right.set_ylim(-0.5e06, 3.5e06)

    def ticks_including_boundary(lo, hi, nbins, boundary):
        base = MaxNLocator(nbins=nbins).tick_values(lo, hi)
        base = base[(base >= lo) & (base <= hi)]
        if not np.any(np.isclose(base, boundary)):
            base = np.sort(np.append(base, boundary))
        return base

    def fmt_tick(v):
        return f"{int(v)}" if float(v).is_integer() else f"{v:g}"

    left_ticks = ticks_including_boundary(*x_left, 2, x_left[1])
    right_ticks = ticks_including_boundary(*x_right, 2, x_right[0])
    ax_left.set_xticks(left_ticks)
    ax_left.set_xticklabels([fmt_tick(v) for v in left_ticks])
    right_labels = [fmt_tick(v) for v in right_ticks]
    if np.isclose(right_ticks[0], left_ticks[-1]):
        right_labels[0] = ""
    ax_right.set_xticks(right_ticks)
    ax_right.set_xticklabels(right_labels)

    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.tick_params(axis="y", which="both", length=0)
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)

    for ax in (ax_left, ax_right):
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
    ax_right.yaxis.get_offset_text().set_visible(False)

    ax_left.set_ylabel(r"Shielding TCF / rad$^2$$\cdot$s$^{-2}$$\cdot$T$^{-2}$")
    fig.text(0.45, -0.02, "$t$ / ps", ha="center", va="bottom",
              fontsize=ff.master_formatting.get("axes.labelsize", 18) if HAVE_FF else 14)

    fig.canvas.draw()
    left_pos = ax_left.get_position()
    right_pos = ax_right.get_position()

    break_dx, break_dy, break_spacing = 0.010, 0.035, 0.045
    x_boundary = (left_pos.x1 + right_pos.x0) / 2
    for y_edge in [left_pos.y0, left_pos.y1]:
        for offset in [-break_spacing / 2, break_spacing / 2]:
            xs = [x_boundary + offset - break_dx, x_boundary + offset + break_dx]
            ys = [y_edge - break_dy / 2, y_edge + break_dy / 2]
            line = plt.Line2D(xs, ys, transform=fig.transFigure, color="black",
                               linewidth=1.5, zorder=10, clip_on=False)
            fig.add_artist(line)

    cax = fig.add_axes([right_pos.x1 + 0.04, right_pos.y0, 0.03, right_pos.height])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(config["x_label"])

    proxy = [Line2D([0], [0], color="black", linestyle=SERIES_LINESTYLES[i], linewidth=2)
             for i in range(len(config["subcase_order"]))]
    labels = [config["subcase_display"][sc] for sc in config["subcase_order"]]
    ax_right.legend(proxy, labels, frameon=False, loc="upper right")

    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"Wrote {out_path} ({n_plotted} curves)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--x-left", type=float, nargs=2, default=[0.0, 5.0],
                     help="time range (ps) shown in the left (short-time zoom) panel")
    ap.add_argument("--x-right", type=float, nargs=2, default=[5.0, 100.0],
                     help="time range (ps) shown in the right (longer tail) panel")
    ap.add_argument("--case", type=int, choices=[1, 2, 3], default=None,
                     help="only plot this case, default: all 3")
    ap.add_argument("--n-angles", type=int, default=10,
                     help="must match the locked angle grid used in step2f_powder_averaged_relaxation.py")
    ap.add_argument("--gamma-xe", type=float, default=GAMMA_XE_DEFAULT)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    theta_grid_deg = list(np.linspace(0.0, 90.0, args.n_angles))
    cases = [args.case] if args.case else [1, 2, 3]

    for case in cases:
        config = CASE_CONFIG[case]
        cdf = manifest[manifest["case"] == case]
        if cdf.empty:
            print(f"No rows for case {case}, skipping")
            continue

        val_to_color, cmap = variable_colors(cdf["variable_value"].tolist())
        vmin, vmax = min(val_to_color), max(val_to_color)

        print(f"Case {case}: building TCFs for {len(cdf)} systems at {args.n_angles} angles each...")
        tcf_cache = {}
        for _, row in cdf.iterrows():
            run_id = row["run_id"]
            assembled_path = args.results_dir / run_id / f"{run_id}_assembled.npz"
            if not assembled_path.exists():
                continue
            tcf_cache[run_id] = build_tcf_all_angles(assembled_path, args.gamma_xe, theta_grid_deg)
            print(f"  {run_id} done")

        for theta_deg in theta_grid_deg:
            out_path = args.results_dir / f"case{case}_tcf_theta{theta_deg:02.0f}.svg"
            plot_case_tcf_theta(cdf, config, theta_deg, tcf_cache, val_to_color, cmap, vmin, vmax,
                                 args.x_left, args.x_right, out_path)


if __name__ == "__main__":
    main()
