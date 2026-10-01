#!/usr/bin/env python3
"""
T1 vs the swept variable (temperature/diameter/loading), one SVG per case,
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
they remain in results_summary.csv (T1_old_s / T1_old_sem_s) alongside the
new ones for direct comparison, this script only plots the new, primary,
reported result.

Horizontal grey dashed lines mark
Hanni et al.'s experimental gas-phase T1 at each field (direct value at
14.1 T, B0^2-scaled from their 8.0 T value at 9.4 T, see the constants
below for exact sourcing).

Writes 3 SVGs directly into --results-dir: case1_T1.svg, case2_T1.svg,
case3_T1.svg.

Usage
-----
    python plot_T1_all_cases.py \\
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

SERIES_MARKERS = ["o", "s"]
FIELD_LINESTYLES = ["-", "--"]

# Hanni, Lantto, Vaara 2011 PCCP, Table 1. CALCULATED (theory) values, not
# experimental ones, see the conversation record for why: footnote 40's B0^2
# scaling applies to their own theory (evaluated fresh at any field from the
# actual simulated SDF), not to raw measurements, which mix in non-CSA
# mechanisms (their own introduction: spin-rotation dominates gas-phase Xe
# relaxation at low B0). Two conditions, matching Figure 2's choice of 99.8
# amg for the dense case:
#   ~1 amg (dilute limit):  direct calc. (Expon. fit) at 8.0 T (69.4e3 s) AND
#                            at 14.1 T (22.5e3 s), two independent numbers.
#                            B0^2-scaling each to 9.4 T agrees to within 0.7%
#                            of the other (50.3e3 vs 50.6e3 s), the paper's
#                            own data confirming the scaling assumption here;
#                            the average, 50.4e3 s, is used at 9.4 T.
#   99.8 amg:                direct calc. (Plateau fit) at 9.4 T (4.4e3 s),
#                            B0^2-scaled to 14.1 T (1.96e3 s).
# No experimental T2 is reported anywhere in the paper for gas-phase Xe, so
# no reference line is added to the T2 plots.
_HANNI_T1_CALC_REFERENCES = [
    # (source_label, color, field_T, value_s, is_direct)
    ("Xe(g), ~1 amg",   "0.5",     9.4,  50.4e3, False),
    ("Xe(g), ~1 amg",   "0.5",     14.1, 22.5e3, True),
    ("Xe(g), 99.8 amg", "#8c564b", 9.4,  4.4e3,  True),
    ("Xe(g), 99.8 amg", "#8c564b", 14.1, 1.96e3, False),
]

# Experimental (raw measurement) T1 values from the same Table 1, used only
# where directly reported at one of our two fields, never scaled: scaling is
# only valid for the CSA-only calculated values above (see the conversation
# record, spin-rotation contaminates the raw measurement, which does not
# follow the same B0^2 law). Of the four (condition, field) combinations,
# only two have a directly reported experimental value at all:
#   ~1 amg (dilute limit): expt. 22.5e3 +/- 4.0e3 s at 14.1 T only
#                            (Anger et al., Phys. Rev. A 2008, 78, 043406,
#                            as tabulated in Hanni et al. Table 1 footnote f)
#   99.8 amg:               expt. 3.6e3 +/- 0.4e3 s at 9.4 T only
#                            (reported directly in Hanni et al. Table 1)
# Colors are deliberately distinct from both the calculated-reference colors
# above and the LL/HL and (10,0)/(30,0) data-series colors used elsewhere in
# this plot.
_EXPT_T1_REFERENCES = [
    # (source_label, color, field_T, value_s, sem_s, alpha)
    ("Xe(g), ~1 amg",   "#1f77b4", 14.1, 22.5e3, 4.0e3, 0.4),
    ("Xe(g), 99.8 amg", "orange", 9.4,  3.6e3,  0.4e3, 1.0),
]
REF_LINESTYLE_BY_FIELD = {8.0: (0, (1, 1)), 9.4: (0, (6, 3)), 14.1: (0, (2, 2))}
# Deliberately a different dash pattern from any calculated-reference
# linestyle above, since the ~1 amg calculated and experimental values at
# 14.1 T coincide almost exactly (22.5 vs 22.5 +/- 4.0 e3 s), same field
# means REF_LINESTYLE_BY_FIELD would otherwise give them the identical
# pattern, hiding one line completely behind the other.
EXPT_LINESTYLE = '-'#(0, (4, 1, 1, 1))

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

QUANTITY = "T1"
Y_LABEL = r"$T_1$ / $10^3$ s"
Y_SCALE = 1.0e-3  # display in units of 10^3 s, matching Hanni et al. Table 1's own convention


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

    for label_source, color, field, value_s, is_direct in _HANNI_T1_CALC_REFERENCES:
        ls = REF_LINESTYLE_BY_FIELD[field]
        ax.axhline(value_s * Y_SCALE, color=color, linestyle=ls, linewidth=1.8, zorder=0)
        note = "calc." if is_direct else "calc. scaled"
        legend_handles.append(Line2D([0], [0], color=color, linestyle=ls, linewidth=1.8))
        legend_labels.append(f"{note} {label_source}, {field:g} T")

    for label_source, color, field, value_s, sem_s, alpha in _EXPT_T1_REFERENCES:
        ax.axhline(value_s * Y_SCALE, color=color, linestyle=EXPT_LINESTYLE, linewidth=1.8,
                   alpha=alpha, zorder=1)
        legend_handles.append(Line2D([0], [0], color=color, linestyle=EXPT_LINESTYLE,
                                      linewidth=1.8, alpha=alpha))
        legend_labels.append(f"expt. {label_source}, {field:g} T")

    ax.set_xlabel(config["x_label"])
    ax.set_ylabel(Y_LABEL)
    ax.set_yscale("log")

    ax.set_ylim(1e-01, 1e02)

    ax.legend(legend_handles, legend_labels, frameon=False, fontsize=13,
              loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

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
        out_path = args.results_dir / f"case{case}_T1.svg"
        plot_case(manifest, case, args.results_dir, args.b0_fields, out_path)


if __name__ == "__main__":
    main()
