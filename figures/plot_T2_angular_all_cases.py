#!/usr/bin/env python3
"""
T2 vs theta (SWCNT-axis-to-B0 tilt angle), all systems of one (case,
subcase) overlaid in one plot, for the ESI. Companion to
plot_T2_all_cases.py (which plots the powder-averaged T1 vs the swept
variable, for the main text) and plot_tcf_all_cases.py (whose RdBu_r
cool-to-hot coloring convention this follows for the swept variable).

Two things need distinguishing on one plot here:
  - the swept variable (temperature/diameter/loading)  -> color, RdBu_r
  - the B0 field                                        -> linestyle, solid/dashed
Subcase (LL/HL, or (10,0)/(30,0)) is not overlaid within a single figure,
each subcase gets its own SVG, so a case that would otherwise carry 28
overlapping curves (7 variable values x 2 subcases x 2 fields) splits into
two, cleaner, 14-curve figures.

Reads results_per_theta.csv directly (built by collect_results.py), not
individual systems' .npz files, since that CSV already has every
(system, B0, theta) row needed in one place.

Writes 6 SVGs directly into --results-dir, one per (case, subcase):
case1_T2_angular_low.svg, case1_T2_angular_high.svg,
case2_T2_angular_low.svg, case2_T2_angular_high.svg,
case3_T2_angular_cnt_10_0.svg, case3_T2_angular_cnt_30_0.svg.

Usage
-----
    python plot_T2_angular_all_cases.py \\
        --per-theta-csv ./results_per_theta.csv \\
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

FIELD_LINESTYLES = ["-", "--"]

CASE_CONFIG = {
    1: dict(cbar_label="$T$ / K", subcase_order=["low", "high"],
            subcase_display={"low": "LL", "high": "HL"}),
    2: dict(cbar_label=r"$d$ / $\mathrm{\AA}$", subcase_order=["low", "high"],
            subcase_display={"low": "LL", "high": "HL"}),
    3: dict(cbar_label="Xe loading / % max", subcase_order=["cnt_10_0", "cnt_30_0"],
            subcase_display={"cnt_10_0": "(10,0)", "cnt_30_0": "(30,0)"}),
}

QUANTITY = "T2"
Y_LABEL = r"$T_2$ / s"


def shared_log_ylim(df: pd.DataFrame):
    """Y-limits computed from BOTH T1_s and T2_s columns, not just the
    quantity this script plots, rounded outward to the nearest power of
    10. plot_T2_angular_all_cases.py and plot_T2_angular_all_cases.py both
    run this same function on the same CSV, so they arrive at identical
    limits independently, no shared constant or coordinated run needed.
    Log-scale tick positions are a deterministic function of the axis
    range, so identical limits also give identical tick counts for free."""
    vals = pd.concat([df["T1_s"], df["T2_s"]]).dropna()
    vals = vals[vals > 0]
    lo = 10 ** np.floor(np.log10(vals.min()))
    hi = 10 ** np.ceil(np.log10(vals.max()))
    return lo, hi


def variable_colors(values):
    """Same RdBu_r cool-to-hot convention as plot_tcf_all_cases.py."""
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


def plot_case_subcase(df: pd.DataFrame, case: int, subcase: str, b0_fields, ylim, out_path: Path):
    config = CASE_CONFIG[case]
    cdf = df[(df["case"] == case) & (df["subcase"] == subcase)]
    if cdf.empty:
        print(f"No rows for case {case}, subcase {subcase}, skipping")
        return

    val_to_color, cmap = variable_colors(cdf["variable_value"].tolist())
    vmin, vmax = min(val_to_color), max(val_to_color)

    if HAVE_FF:
        fig, ax = ff.make_figure(figsize=(6.0, 5.0), ax_box=(0.20, 0.18, 0.58, 0.72))
    else:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        fig.subplots_adjust(left=0.30, bottom=0.18, right=0.78, top=0.90)
    cax = fig.add_axes([0.82, 0.18, 0.03, 0.72])

    n_curves = 0
    for run_id, g_run in cdf.groupby("run_id"):
        color = val_to_color[g_run["variable_value"].iloc[0]]
        for fi, B0 in enumerate(b0_fields):
            g = g_run[np.isclose(g_run["B0_T"], B0)].sort_values("theta_deg")
            if g.empty:
                continue
            ax.plot(g["theta_deg"], g[f"{QUANTITY}_s"], color=color,
                    linestyle=FIELD_LINESTYLES[fi], marker="o",
                    markersize=5, linewidth=1.5)
            n_curves += 1

    ax.set_xlabel(r"$\theta$ / deg")
    ax.set_ylabel(Y_LABEL)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0))
    ax.set_xlim(0, 90)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(config["cbar_label"])

    field_proxy = [Line2D([0], [0], color="black", linestyle=FIELD_LINESTYLES[i], linewidth=1.5)
                   for i in range(len(b0_fields))]
    field_labels = [f"{B0:g} T" for B0 in b0_fields]
    ax.legend(field_proxy, field_labels, frameon=False, loc="lower right", fontsize=14)

    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"Wrote {out_path} ({n_curves} curves)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-theta-csv", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--b0-fields", type=float, nargs="+", default=[9.4, 14.1])
    args = ap.parse_args()

    df = pd.read_csv(args.per_theta_csv)
    ylim = shared_log_ylim(df)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for case in [1, 2, 3]:
        for subcase in CASE_CONFIG[case]["subcase_order"]:
            out_path = args.results_dir / f"case{case}_T2_angular_{subcase}.svg"
            plot_case_subcase(df, case, subcase, args.b0_fields, ylim, out_path)


if __name__ == "__main__":
    main()
