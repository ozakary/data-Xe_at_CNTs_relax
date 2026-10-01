#!/usr/bin/env python3
"""
Build the run manifest for the three sweeps (temperature, diameter, Xe
loading). Encodes exactly the paths and variables described for the letter's
extended dataset. Edit the lists in build_manifest() if systems are added or
removed later, everything downstream (run_all_systems.py, collect_results.py,
plot_sweeps.py) reads the manifest, not this file's logic.

Usage
-----
    python build_run_manifest.py --root . --systems-info ./cnt_systems_info.csv --out ./run_manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_manifest():
    rows = []

    # --- Case 1: temperature sweep, fixed (22,0), two loading tiers ---
    temps = [100, 150, 200, 250, 300, 350, 400]
    for loading_label, n_xe in [("low", 13), ("high", 53)]:
        for T in temps:
            data_dir = f"./fixed_density/changing_temperature/n_0/cnt_22_0_u23_Xe{n_xe}/{T}K_10ns"
            rows.append(dict(
                case=1, case_name="temperature", subcase=loading_label,
                data_dir=data_dir, system_name=f"cnt_22_0_u23_Xe{n_xe}_{T}K",
                chirality="(22,0)", n_xe=n_xe, temperature_K=T,
                variable_name="temperature_K", variable_value=T,
                geom_key="cnt_22_0_u23_Xe%d.xyz" % n_xe,
            ))

    # --- Case 2: diameter sweep, fixed density tier, two loading tiers ---
    low = [("10_0", 4), ("14_0", 4), ("18_0", 9), ("22_0", 13), ("26_0", 13), ("30_0", 13)]
    high = [("10_0", 18), ("14_0", 18), ("18_0", 35), ("22_0", 53), ("26_0", 53), ("30_0", 53)]
    for loading_label, pairs in [("low", low), ("high", high)]:
        for chir, n_xe in pairs:
            data_dir = f"./fixed_density/changing_size/n_0/cnt_{chir}_u23_Xe{n_xe}_10ns"
            rows.append(dict(
                case=2, case_name="diameter", subcase=loading_label,
                data_dir=data_dir, system_name=f"cnt_{chir}_u23_Xe{n_xe}",
                chirality=f"({chir.replace('_', ',')})", n_xe=n_xe, temperature_K=300,
                variable_name="diameter_A", variable_value=None,  # filled in after geometry join
                geom_key="cnt_%s_u23_Xe%d.xyz" % (chir, n_xe),
            ))

    # --- Case 3: Xe loading sweep, fixed diameter, two diameters ---
    loadings_10_0 = [4, 7, 10, 12, 15, 18]
    loadings_30_0 = [13, 21, 29, 37, 45, 53]
    for chir, loadings in [("10_0", loadings_10_0), ("30_0", loadings_30_0)]:
        for n_xe in loadings:
            data_dir = f"./changing_density/n_0/cnt_{chir}_u23_Xe{n_xe}_10ns"
            rows.append(dict(
                case=3, case_name="loading", subcase=f"cnt_{chir}",
                data_dir=data_dir, system_name=f"cnt_{chir}_u23_Xe{n_xe}",
                chirality=f"({chir.replace('_', ',')})", n_xe=n_xe, temperature_K=300,
                variable_name="loading_pct", variable_value=None,  # filled in after geometry join
                geom_key="cnt_%s_u23_Xe%d.xyz" % (chir, n_xe),
            ))

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("."), help="root directory the data_dir paths are relative to")
    ap.add_argument("--systems-info", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    manifest = build_manifest()
    geom = pd.read_csv(args.systems_info)
    geom = geom.drop_duplicates(subset="filename")  # some filenames repeat across the pasted table sections

    manifest = manifest.merge(geom, left_on="geom_key", right_on="filename", how="left", suffixes=("", "_geom"))
    missing = manifest[manifest["filename"].isna()]
    if len(missing):
        print("WARNING: no geometry match found for these systems (check cnt_systems_info.csv):")
        print(missing[["system_name", "geom_key"]].to_string(index=False))

    manifest.loc[manifest["variable_name"] == "diameter_A", "variable_value"] = manifest["diameter_A"]
    manifest["loading_pct_computed"] = 100.0 * manifest["n_xe"] / manifest["geometric_max_Xe"]
    manifest.loc[manifest["variable_name"] == "loading_pct", "variable_value"] = manifest["loading_pct_computed"]

    manifest["data_dir_full"] = manifest["data_dir"].apply(lambda p: str((args.root / p).resolve()))

    # system_name alone collides across cases (e.g. cnt_10_0_u23_Xe4 appears both as a case-2
    # diameter-sweep point and a case-3 loading-sweep point, from two DIFFERENT directories,
    # fixed_density/changing_size/... vs changing_density/...). run_id disambiguates storage
    # so results are never silently shared/skipped across cases regardless of whether the
    # underlying trajectories happen to be identical.
    manifest["run_id"] = manifest.apply(lambda r: f"{r['system_name']}__case{r['case']}_{r['subcase']}", axis=1)
    dupes = manifest[manifest.duplicated("run_id", keep=False)]
    if len(dupes):
        print("WARNING: run_id still not unique, check manually:")
        print(dupes[["run_id", "data_dir"]].to_string(index=False))

    manifest.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(manifest)} systems "
          f"(case1={len(manifest[manifest.case==1])}, case2={len(manifest[manifest.case==2])}, "
          f"case3={len(manifest[manifest.case==3])})")


if __name__ == "__main__":
    main()
