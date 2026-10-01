#!/usr/bin/env python3
"""
Prints ready-to-paste LaTeX rows for Tables S7-S9 (T1, T2 at both fields,
powder-averaged over the locked 10-point theta grid, sin(theta)-weighted,
on rates, point from step2f_powder_averaged_relaxation.py's pooled fit,
uncertainty from step4_powder_averaged_per_track_uncertainty.py's
atom-to-atom SEM, plus the orientational spread across theta from the
pooled fit's own T1_min/T1_max, T2_min/T2_max), one block per series
(temperature, diameter, loading), in the same row order used in those
tables. This is the same point+error split already used in
plot_T1_all_cases.py / plot_T2_all_cases.py.

The spread is shown as a single value, $|T_1(\\theta_{\\max})-T_1(\\theta_{\\min})|$
(and likewise for $T_2$), in one column next to each quantity's point +/-
SEM column, rather than as separate min and max columns. This spread is a
distinct quantity from the SEM, the real orientational anisotropy of the
system's own T1(theta)/T2(theta), not a statistical uncertainty on the
powder average, matching the text in Section II.4 of the ESI.

Usage
-----
    python print_t1_t2_summary_table.py \\
        --manifest ./run_manifest.csv \\
        --results-dir ./results \\
        --b0-fields 9.4 14.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CASE_CONFIG = {
    1: dict(var_label="$T$ / K", subcase_order=["low", "high"],
            subcase_display={"low": "LL", "high": "HL"}),
    2: dict(var_label="Tube", subcase_order=["low", "high"],
            subcase_display={"low": "LL", "high": "HL"}),
    3: dict(var_label="Loading / \\%", subcase_order=["cnt_10_0", "cnt_30_0"],
            subcase_display={"cnt_10_0": "(10,0)", "cnt_30_0": "(30,0)"}),
}


def fmt_val_sem(value, sem, sig=4):
    return f"{value:.{sig}g} $\\pm$ {sem:.2g}"


def fmt_spread(lo, hi, sig=3):
    return f"{abs(hi - lo):.{sig}g}"


def get_row(run_id: str, results_dir: Path, var_display: str, loading_display: str, b0_fields):
    sys_dir = results_dir / run_id
    powder_path = sys_dir / f"{run_id}_powder_relaxation.npz"
    pt_path = sys_dir / f"{run_id}_powder_relaxation_per_track.npz"
    if not (powder_path.exists() and pt_path.exists()):
        print(f"  % MISSING DATA for {run_id}")
        return None

    d2f = np.load(powder_path, allow_pickle=True)
    powder = d2f["powder"].item()
    d4 = np.load(pt_path, allow_pickle=True)
    summary = d4["summary"].item()

    cells = []
    for B0 in b0_fields:
        if B0 not in powder or B0 not in summary:
            cells.append("--")
            cells.append("--")
            cells.append("--")
            cells.append("--")
            continue
        cells.append(fmt_val_sem(powder[B0]["T1"], summary[B0]["T1_sem"]))
        cells.append(fmt_spread(powder[B0]["T1_min"], powder[B0]["T1_max"]))
        cells.append(fmt_val_sem(powder[B0]["T2"], summary[B0]["T2_sem"]))
        cells.append(fmt_spread(powder[B0]["T2_min"], powder[B0]["T2_max"]))

    return f"    {loading_display} & {var_display} & " + " & ".join(cells) + " \\\\"


def print_case(manifest: pd.DataFrame, case: int, results_dir: Path, b0_fields):
    config = CASE_CONFIG[case]
    cdf = manifest[manifest["case"] == case]
    print(f"% --- Case {case} ({config['var_label']}) ---")
    for subcase in config["subcase_order"]:
        sub = cdf[cdf["subcase"] == subcase].sort_values("variable_value")
        loading_display = config["subcase_display"][subcase]
        for _, row in sub.iterrows():
            if case == 2:
                var_display = row["chirality"]
            elif case == 3:
                var_display = f"{row['variable_value']:.1f}"
            else:
                var_display = f"{row['variable_value']:g}"
            line = get_row(row["run_id"], results_dir, var_display, loading_display, b0_fields)
            if line:
                print(line)
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--b0-fields", type=float, nargs="+", default=[9.4, 14.1])
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    for case in [1, 2, 3]:
        print_case(manifest, case, args.results_dir, args.b0_fields)


if __name__ == "__main__":
    main()
