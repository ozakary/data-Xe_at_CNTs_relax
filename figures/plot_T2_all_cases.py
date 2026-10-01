#!/usr/bin/env python3
"""
T2 vs the swept variable (temperature/diameter/loading), one SVG per case,
following code_plot_delta_iso_temp_sys-shift.py / _size_sys-shift.py's
errorbar style and colors (#264653/#2a9d8f, markers o/s) exactly, minus
the "(shifted)" comparison curves (not applicable to a relaxation time) and
minus the axhline(0) (T1 is strictly positive). Since both B0 fields need
to appear in one file (not one file per field), color carries the series
(LL/HL or (10,0)/(30,0)) and linestyle carries the field, solid for the
first field, dashed for the second.

Reads the powder-averaged methodology's output
(step2f_powder_averaged_relaxation.py / step4_powder_averaged_per_track_uncertainty.py),
not the old theta=0-only files. The plotted point is the pooled TCF's
sin(theta)-weighted powder average across the 10-angle grid (the lowest-
noise estimate, Step 2f), the error bar is the atom-to-atom SEM of each
tracked atom's own powder-averaged rate (Step 4-new), mirroring exactly the
same point-estimate/uncertainty split this project has used from the
start, just applied to the new methodology's own pooled and per-track
outputs instead of the theta=0 ones. The old theta=0 numbers are not lost,
they remain in results_summary.csv (T2_old_s / T2_old_sem_s) alongside the
new ones for direct comparison, this script only plots the new, primary,
reported result.

Writes 3 SVGs directly into --results-dir: case1_T1.svg, case2_T1.svg,
case3_T1.svg.

Usage
-----
    python plot_T2_all_cases.py \\
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

try:
    import figure_formatting_v2 as ff
    ff.set_rcParams(ff.master_formatting)
    HAVE_FF = True
except ImportError:
    print("figure_formatting_v2 not found, using default matplotlib settings.")
    HAVE_FF = False

SERIES_MARKERS = ["o", "s"]
FIELD_LINESTYLES = ["-", "--"]

CASE_CONFIG = {
    1: dict(x_label="$T$ / K", subcase_order=["low", "high"],
            subcase_display={"low": "LL", "high": "HL"},
            colors=["#264653", "#2a9d8f"]),
    2: dict(x_label=r"$d$ / $\mathrm{\AA}$", subcase_order=["low", "high"],
            subcase_display={"low": "LL", "high": "HL"},
            colors=["#264653", "#2a9d8f"]),
    3: dict(x_label="Xe loading / % max", subcase_order=["cnt_10_0", "cnt_30_0"],
            subcase_display={"cnt_10_0": "(10,0)", "cnt_30_0": "(30,0)"},
            colors=["#3A86FF", "#FF006E"]),
}

QUANTITY = "T2"
Y_LABEL = r"$T_2$ / $10^3$ s"
Y_SCALE = 1.0e-3  # display in units of 10^3 s, matching plot_T1_all_cases.py's convention


def plot_case(manifest: pd.DataFrame, case: int, results_dir: Path, b0_fields, out_path: Path):
    config = CASE_CONFIG[case]
    cdf = manifest[manifest["case"] == case]
    if cdf.empty:
        print(f"No rows for case {case}, skipping")
        return

    if HAVE_FF:
        fig, ax = ff.make_figure(preset=1)
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.subplots_adjust(left=0.22, bottom=0.18, right=0.92, top=0.90)

    legend_handles, legend_labels = [], []
    for si, subcase in enumerate(config["subcase_order"]):
        sub = cdf[cdf["subcase"] == subcase].sort_values("variable_value")
        color = config["colors"][si]
        marker = SERIES_MARKERS[si]
        for fi, B0 in enumerate(b0_fields):
            xs, ys, yerrs = [], [], []
            for _, row in sub.iterrows():
                powder_path = results_dir / row["run_id"] / f"{row['run_id']}_powder_relaxation.npz"
                pt_path = results_dir / row["run_id"] / f"{row['run_id']}_powder_relaxation_per_track.npz"
                if not (powder_path.exists() and pt_path.exists()):
                    continue
                d2f = np.load(powder_path, allow_pickle=True)
                powder = d2f["powder"].item()
                if B0 not in powder:
                    continue
                d4 = np.load(pt_path, allow_pickle=True)
                summary = d4["summary"].item()
                if B0 not in summary:
                    continue
                xs.append(row["variable_value"])
                ys.append(powder[B0][QUANTITY] * Y_SCALE)
                yerrs.append(summary[B0][f"{QUANTITY}_sem"] * Y_SCALE)
            if not xs:
                continue
            h = ax.errorbar(xs, ys, yerr=yerrs, marker=marker, color=color,
                             linestyle=FIELD_LINESTYLES[fi], markersize=6, linewidth=2,
                             capsize=4, capthick=2)
            legend_handles.append(h)
            legend_labels.append(f"{config['subcase_display'][subcase]}, {B0:g} T")

    ax.set_xlabel(config["x_label"])
    ax.set_ylabel(Y_LABEL)
    ax.set_yscale("log")

    ax.set_ylim(1e-01, 1e01)
    
#    ax.legend(legend_handles, legend_labels, frameon=False, fontsize=13,
#              loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    fig.savefig(out_path, format="svg")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--b0-fields", type=float, nargs="+", default=[9.4, 14.1])
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    for case in [1, 2, 3]:
        out_path = args.results_dir / f"case{case}_T2.svg"
        plot_case(manifest, case, args.results_dir, args.b0_fields, out_path)


if __name__ == "__main__":
    main()
