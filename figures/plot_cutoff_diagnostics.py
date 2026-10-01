#!/usr/bin/env python3
"""
Per-system fit-cutoff robustness plots for the ESI, T1 and T2 vs cutoff,
at a single field (default 9.4 T), for the powder-averaged methodology.
No variance/deviation-from-reference plots, unlike
step6_check_cutoff_robustness_powder.py's console summary, just the two
plain quantity-vs-cutoff plots, properly formatted and in SVG.

Reads the already-computed --detailed CSV from
step6_check_cutoff_robustness_powder.py, does not refit or recompute
anything. That CSV's cutoff_ps/T1_s/T2_s columns happen to share the exact
same names as the old (theta=0-only) check_cutoff_robustness.py's output,
so the plotting logic below is unchanged from the original version of this
script, only the docstring, expected input, and output filenames differ.

For every system present in --detailed at the requested field, writes two
SVGs into its own results/<run_id>/ folder:
  {run_id}_cutoff_T1_powder.svg
  {run_id}_cutoff_T2_powder.svg
Named distinctly from the old {run_id}_cutoff_T1.svg / _T2.svg (theta=0
only) diagnostics, which this script does not touch, so both remain
available side by side for comparison, matching how the rest of this
project's old and new pipelines were kept side by side throughout.

Usage
-----
    python plot_cutoff_diagnostics.py \\
        --detailed ./powder_cutoff_robustness_detailed.csv \\
        --results-dir ./results \\
        --b0-field 9.4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import figure_formatting_v2 as ff
    ff.set_rcParams(ff.master_formatting)
    HAVE_FF = True
except ImportError:
    print("figure_formatting_v2 not found, using default matplotlib settings.")
    HAVE_FF = False


QUANTITY_COLORS = {"T1": "#264653", "T2": "#E76F51"}


def plot_one(df_sys: pd.DataFrame, quantity: str, y_label: str, out_path: Path):
    if HAVE_FF:
        fig, ax = ff.make_figure(preset=1)
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.subplots_adjust(left=0.22, bottom=0.18, right=0.92, top=0.90)

    g = df_sys.sort_values("cutoff_ps")
    ax.plot(g["cutoff_ps"], g[f"{quantity}_s"], marker="o", ms=6, lw=2, color=QUANTITY_COLORS[quantity])

    ax.set_xscale("log")
    ax.set_xlabel("Fit cutoff / ps")
    ax.set_ylabel(y_label)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detailed", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--b0-field", type=float, default=9.4)
    args = ap.parse_args()

    detailed = pd.read_csv(args.detailed)
    detailed = detailed[np.isclose(detailed["B0_T"], args.b0_field)]
    if detailed.empty:
        print(f"No rows at B0={args.b0_field} T in {args.detailed}")
        return

    for run_id, df_sys in detailed.groupby("run_id"):
        sys_dir = args.results_dir / run_id
        sys_dir.mkdir(parents=True, exist_ok=True)
        plot_one(df_sys, "T1", r"$T_1$ / s", sys_dir / f"{run_id}_cutoff_T1_powder.svg")
        plot_one(df_sys, "T2", r"$T_2$ / s", sys_dir / f"{run_id}_cutoff_T2_powder.svg")


if __name__ == "__main__":
    main()
