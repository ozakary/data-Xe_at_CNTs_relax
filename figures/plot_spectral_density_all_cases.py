#!/usr/bin/env python3
"""
Spectral density J(omega) vs omega, one SVG per (case, theta), all systems
of each case overlaid, same RdBu_r cool-to-hot coloring and solid/dashed
series convention as plot_tcf_all_cases.py. J(omega) is reconstructed from
each system's saved bi/mono-exponential fit AT THAT ANGLE (Step 2f's
per_theta[theta_deg]["amps_ps"/"taus_ps"]), a sum of Lorentzians, evaluated
on a log-spaced omega grid, not recomputed from the raw TCF. Axis labels
follow Hanni, Lantto, Vaara 2011's wording ("shielding spectral density
function"), in our own actual units. The two B0 fields' Larmor frequencies
are marked as thin vertical guides.

This is the theta-resolved companion to the old (theta=0-only)
plot_spectral_density_all_cases.py: reads step2f_powder_averaged_relaxation.py's
output directly, which already has the fit parameters at every angle in
the locked 10-point grid saved, no tensor rebuild needed (unlike the TCF
version, SDF only needs the already-fit amps/taus, not the raw
correlation function itself).

Writes 30 SVGs directly into --results-dir: case{1,2,3}_sdf_theta{00,10,
...,90}.svg. theta=00 is the one intended for the main text (SWCNT axis
parallel to B0, matching the pre-existing single-angle figure this project
started with), the other nine are ESI-only.

Usage
-----
    python plot_spectral_density_all_cases.py \\
        --manifest ./run_manifest.csv \\
        --results-dir ./results \\
        --b0-fields 9.4 14.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    import figure_formatting_v2 as ff
    ff.set_rcParams(ff.master_formatting)
    HAVE_FF = True
except ImportError:
    print("figure_formatting_v2 not found, using default matplotlib settings.")
    HAVE_FF = False

SERIES_LINESTYLES = ["-", "--"]

CASE_CONFIG = {
    1: dict(x_label="$T$ / K", subcase_order=["low", "high"],
            subcase_display={"low": "LL", "high": "HL"}),
    2: dict(x_label=r"$d$ / $\mathrm{\AA}$", subcase_order=["low", "high"],
            subcase_display={"low": "LL", "high": "HL"}),
    3: dict(x_label="Xe loading / % max", subcase_order=["cnt_10_0", "cnt_30_0"],
            subcase_display={"cnt_10_0": "(10,0)", "cnt_30_0": "(30,0)"}),
}


def variable_colors(values):
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


def lorentzian_sum(omega, amps, taus_s):
    omega = np.atleast_1d(omega).astype(float)
    J = np.zeros_like(omega)
    for A, tau in zip(amps, taus_s):
        J += A * tau / (1.0 + (omega * tau) ** 2)
    return J


def plot_case_sdf_theta(manifest: pd.DataFrame, case: int, theta_deg: float,
                         results_dir: Path, b0_fields, out_path: Path):
    config = CASE_CONFIG[case]
    cdf = manifest[manifest["case"] == case]
    if cdf.empty:
        print(f"No rows for case {case}, skipping")
        return

    val_to_color, cmap = variable_colors(cdf["variable_value"].tolist())
    vmin, vmax = min(val_to_color), max(val_to_color)
    w_grid = np.logspace(6, 13, 400)

    if HAVE_FF:
        fig, ax = ff.make_figure(figsize=(6.0, 5.0), ax_box=(0.20, 0.18, 0.58, 0.72))
    else:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        fig.subplots_adjust(left=0.30, bottom=0.18, right=0.78, top=0.90)
    cax = fig.add_axes([0.82, 0.18, 0.03, 0.72])

    gamma_xe = None
    n_plotted = 0
    for _, row in cdf.iterrows():
        run_id = row["run_id"]
        powder_path = results_dir / run_id / f"{run_id}_powder_relaxation.npz"
        if not powder_path.exists():
            continue
        d = np.load(powder_path, allow_pickle=True)
        per_theta = d["per_theta"].item()
        if theta_deg not in per_theta:
            continue
        amps = list(per_theta[theta_deg]["amps_ps"])
        taus_ps = list(per_theta[theta_deg]["taus_ps"])
        taus_s = [t * 1e-12 for t in taus_ps]
        J = lorentzian_sum(w_grid, amps, taus_s)
        subcase_idx = config["subcase_order"].index(row["subcase"])
        ax.plot(w_grid, J, color=val_to_color[row["variable_value"]],
                 linestyle=SERIES_LINESTYLES[subcase_idx], linewidth=2)
        n_plotted += 1
        if gamma_xe is None:
            gamma_xe = float(d["gamma_xe"])

    if gamma_xe is not None:
        for B0 in b0_fields:
            w0 = abs(gamma_xe) * B0
            ax.axvline(w0, color="gray", lw=2, ls="-", zorder=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\omega$ / rad$\cdot$s$^{-1}$")
    ax.set_ylabel(r"Shielding SDF / rad$^2$$\cdot$s$^{-1}$$\cdot$T$^{-2}$")
    ax.yaxis.set_label_coords(-0.3, 0.42)

    ax.set_xlim(1e06, 2e13)
    ax.set_ylim(5e-09, 1e-05)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(config["x_label"])

    proxy = [Line2D([0], [0], color="black", linestyle=SERIES_LINESTYLES[i], linewidth=2)
             for i in range(len(config["subcase_order"]))]
    labels = [config["subcase_display"][sc] for sc in config["subcase_order"]]
    ax.legend(proxy, labels, frameon=False, loc="lower left")

    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"Wrote {out_path} ({n_plotted} curves)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--b0-fields", type=float, nargs="+", default=[9.4, 14.1])
    ap.add_argument("--n-angles", type=int, default=10,
                     help="must match the locked angle grid used in step2f_powder_averaged_relaxation.py")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    theta_grid_deg = list(np.linspace(0.0, 90.0, args.n_angles))
    for case in [1, 2, 3]:
        for theta_deg in theta_grid_deg:
            out_path = args.results_dir / f"case{case}_sdf_theta{theta_deg:02.0f}.svg"
            plot_case_sdf_theta(manifest, case, theta_deg, args.results_dir, args.b0_fields, out_path)


if __name__ == "__main__":
    main()
