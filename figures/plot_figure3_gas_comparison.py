#!/usr/bin/env python3
"""
Figure 3: bar chart comparing this work's Xe@SWCNT T1 against Hanni et al.
2011's experimental free-gas T1, at four density-matched pairs (this
work's system density in amagat, computed from pore geometry and n_Xe,
versus Hanni et al.'s nearest reported experimental density), all at
B0=9.4 T, ascending by density.

Each density pair gets its own color (RdBu_r, cool-to-hot by density,
the same convention used for the swept-variable coloring throughout this
project's other figures), calc. (this work) solid and full opacity,
expt. (Hanni et al.) hatched and reduced opacity, same hue. No x-axis
ticks or labels, every pairing and its density match is stated in the
legend instead.

Reads all values from figure3_data.csv, one row per bar. Each row's
"label" is used as its own legend text directly (no suffix generation),
"density_amg" is that row's own real density (this work's computed value
for calc. rows, Hanni et al.'s reported value for expt. rows), and "pair"
groups the two rows of one comparison for coloring and bar placement,
ordered ascending by the calc. side's density.

Usage
-----
    python plot_figure3_gas_comparison.py \\
        --csv ./figure3_data.csv \\
        --out ./figure3.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    import figure_formatting_v2 as ff
    ff.set_rcParams(ff.master_formatting)
    HAVE_FF = True
except ImportError:
    print("figure_formatting_v2 not found, using default matplotlib settings.")
    HAVE_FF = False

SOURCE_HATCH = {"calc": None, "expt": "//"}
SOURCE_SHADE = {"calc": 0.0, "expt": 0.40}  # 0 = base hue, >0 = blended toward white
SOURCE_LABELS = {"calc": "calc. (Xe@SWCNT)", "expt": "expt. (Xe(g))"}


def lighten(rgb, amount):
    """Blend an RGB(A) color toward white by the given fraction (0=no
    change, 1=white), used to give calc./expt. of the same density pair
    distinct shades of one hue rather than distinguishing them by alpha."""
    r, g, b = rgb[0], rgb[1], rgb[2]
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount, 1.0)


def density_colors(density_values):
    """RdBu_r, cool (low density) to hot (high density), same convention
    as the swept-variable coloring in plot_tcf_all_cases.py etc."""
    unique_vals = sorted(set(density_values))
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
    return {v: cmap(color_positions[i]) for i, v in enumerate(unique_vals)}


Y_SCALE = 1.0e-3  # display in units of 10^3 s


def plot_figure3(csv_path: Path, out_path: Path):
    df = pd.read_csv(csv_path)
    # order pairs by their calc.-side density, ascending
    pair_order = (df[df["source"] == "calc"]
                  .sort_values("density_amg")["pair"].tolist())
    n_pairs = len(pair_order)
    bar_width = 0.35
    x_positions = np.arange(n_pairs)

    density_by_pair = {p: df[(df["pair"] == p) & (df["source"] == "calc")]["density_amg"].iloc[0]
                        for p in pair_order}
    base_colors = density_colors(density_by_pair.values())

    if HAVE_FF:
        fig, ax = ff.make_figure(figsize=(8.0, 7.5), ax_box=(0.16, 0.34, 0.80, 0.61))
    else:
        fig, ax = plt.subplots(figsize=(8.5, 8.0))
        fig.subplots_adjust(left=0.16, bottom=0.38, right=0.96, top=0.95)

    bar_colors = {}  # (pair, source) -> shaded color, reused for the legend below
    legend_entries = []  # (color, label), in plotted order
    for i, pair in enumerate(pair_order):
        rows = df[df["pair"] == pair]
        base_color = base_colors[density_by_pair[pair]]
        for source, offset in zip(["calc", "expt"], [-bar_width / 2, bar_width / 2]):
            row = rows[rows["source"] == source].iloc[0]
            color = lighten(base_color, SOURCE_SHADE[source])
            bar_colors[(pair, source)] = color
            legend_entries.append((color, row["label"], SOURCE_HATCH[source]))
            ax.bar(
                x_positions[i] + offset, row["T1_s"] * Y_SCALE, width=bar_width,
                color=color, hatch=SOURCE_HATCH[source],
                edgecolor="black", linewidth=1.5,
                yerr=row["T1_sem_s"] * Y_SCALE, capsize=6,
                error_kw=dict(elinewidth=2.5, ecolor="black"),
            )

    ax.set_ylabel(r"$T_1$ / $10^3$ s")
    ax.set_xticks([])
    ax.set_xlim(-0.6, n_pairs - 0.4)
    ax.set_ylim(0.0, 25.0)

    density_handles = [Patch(facecolor=color, edgecolor="black", hatch=hatch, label=label)
                        for color, label, hatch in legend_entries]
    source_handles = [Patch(facecolor="none", edgecolor="black", hatch=SOURCE_HATCH[s],
                             label=SOURCE_LABELS[s]) for s in ["calc", "expt"]]

    legend = ax.legend(
        handles=source_handles + density_handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.05),
        ncol=2, frameon=False
    )

    fig.savefig(out_path, format="svg", bbox_extra_artists=(legend,), bbox_inches=None)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    plot_figure3(args.csv, args.out)


if __name__ == "__main__":
    main()
