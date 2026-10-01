#!/usr/bin/env python3
"""
Walk the results directory and assemble CSVs from every system that
completed the pipeline, now covering both the old theta=0-only results and
the new powder-averaged ones side by side, so a direct comparison is a
single CSV away rather than a re-analysis.

Four CSVs are written:

  results_summary.csv
    One row per (system, B0). The old theta=0 point estimate/SEM
    (T1_old_s, T1_old_sem_s, ...) and the new powder-averaged point
    estimate/SEM (T1_powder_s, T1_powder_sem_s, ...) side by side, plus the
    orientational spread of the pooled T1(theta), T2(theta) curve
    (T1_theta_min_s, T1_theta_max_s, ...), kept explicitly separate from the
    SEM, exactly as decided for this project: the SEM is atom-to-atom
    statistical uncertainty on the powder average, the min/max range is the
    system's own real anisotropy across orientation, not a statistical
    error bar.

  results_detailed.csv
    One row per (system, B0, track): the new per-track powder-averaged
    T1, T2 (mean R^2 across angles for that track), plus the old per-track
    theta=0 values for the same track where available.

  results_per_theta.csv
    One row per (system, B0, theta): T1(theta), T2(theta), fit R^2, from
    the pooled (lowest-noise) curve. This is the input the planned ESI
    angular-dependence figures need, built now so it does not have to be
    re-derived later.

  results_missing.csv
    Any system missing one or more expected output files, so a partial or
    still-running batch is visible at a glance rather than silently
    producing a shorter CSV.

Usage
-----
    python collect_results.py \\
        --manifest ./run_manifest.csv \\
        --results-dir ./results \\
        --out-summary ./results_summary.csv \\
        --out-detailed ./results_detailed.csv \\
        --out-per-theta ./results_per_theta.csv \\
        --out-missing ./results_missing.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--out-summary", required=True, type=Path)
    ap.add_argument("--out-detailed", required=True, type=Path)
    ap.add_argument("--out-per-theta", required=True, type=Path)
    ap.add_argument("--out-missing", required=True, type=Path)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    summary_rows, detailed_rows, per_theta_rows, missing_rows = [], [], [], []
    n_ok, n_partial = 0, 0

    for _, row in manifest.iterrows():
        name = row["run_id"]
        out_dir = args.results_dir / name
        meta = row.to_dict()

        old_relax_path = out_dir / f"{name}_relaxation.npz"
        old_relax_pt_path = out_dir / f"{name}_relaxation_per_track.npz"
        new_powder_path = out_dir / f"{name}_powder_relaxation.npz"
        new_powder_pt_path = out_dir / f"{name}_powder_relaxation_per_track.npz"

        have_old = old_relax_path.exists() and old_relax_pt_path.exists()
        have_new = new_powder_path.exists() and new_powder_pt_path.exists()

        if not have_new:
            missing_rows.append({**meta, "missing": "powder-averaged (step2f/step4_new) output"})
            n_partial += 1
            continue
        if not have_old:
            missing_rows.append({**meta, "missing": "old theta=0 (step2/3/4) output, powder-averaged still collected"})
            n_partial += 1

        # --- new powder-averaged results (always present at this point) ---
        d_new = np.load(new_powder_path, allow_pickle=True)
        b0_fields = list(d_new["b0_fields"])
        powder = d_new["powder"].item()
        per_theta = d_new["per_theta"].item()
        theta_grid_deg = list(d_new["theta_grid_deg"])

        d_new_pt = np.load(new_powder_pt_path, allow_pickle=True)
        pt_summary = d_new_pt["summary"].item()
        pt_r2 = list(d_new_pt["per_track_r2"])

        # --- old theta=0 results (may be missing) ---
        old_by_b0 = {}
        old_pt_summary = {}
        old_pt_r2 = []
        if have_old:
            d_old = np.load(old_relax_path, allow_pickle=True)
            old_by_b0 = {r["B0"]: r for r in d_old["results"]}
            d_old_pt = np.load(old_relax_pt_path, allow_pickle=True)
            old_pt_summary = d_old_pt["summary"].item()
            old_pt_r2 = list(d_old_pt["per_track_r2"])

        for B0 in b0_fields:
            pw = powder[B0]
            s = pt_summary[B0]
            row_summary = {
                **meta, "B0_T": B0,
                "T1_powder_s": pw["T1"], "T1_powder_sem_s": s["T1_sem"],
                "T2_powder_s": pw["T2"], "T2_powder_sem_s": s["T2_sem"],
                "T1_theta_min_s": pw["T1_min"], "T1_theta_max_s": pw["T1_max"],
                "T2_theta_min_s": pw["T2_min"], "T2_theta_max_s": pw["T2_max"],
                "n_tracks": len(s["T1_per_track_powder"]),
                "n_angles": len(theta_grid_deg),
            }
            if B0 in old_by_b0:
                old_r = old_by_b0[B0]
                old_s = old_pt_summary.get(B0, {})
                row_summary.update({
                    "T1_old_s": old_r["T1"], "T1_old_sem_s": old_s.get("T1_sem"),
                    "T2_old_s": old_r["T2"], "T2_old_sem_s": old_s.get("T2_sem"),
                })
            summary_rows.append(row_summary)

            T1_pt = s["T1_per_track_powder"]
            T2_pt = s["T2_per_track_powder"]
            old_T1_pt = old_pt_summary.get(B0, {}).get("T1_per_track")
            old_T2_pt = old_pt_summary.get(B0, {}).get("T2_per_track")
            for k in range(len(T1_pt)):
                row_detailed = {
                    **meta, "B0_T": B0, "track": k,
                    "T1_powder_s": T1_pt[k], "T2_powder_s": T2_pt[k],
                    "mean_fit_R2_across_angles": pt_r2[k] if k < len(pt_r2) else None,
                }
                if old_T1_pt is not None and k < len(old_T1_pt):
                    row_detailed["T1_old_s"] = old_T1_pt[k]
                    row_detailed["T2_old_s"] = old_T2_pt[k]
                    row_detailed["old_fit_R2"] = old_pt_r2[k] if k < len(old_pt_r2) else None
                detailed_rows.append(row_detailed)

            for theta_deg in theta_grid_deg:
                pt = per_theta[theta_deg]
                per_theta_rows.append({
                    **meta, "B0_T": B0, "theta_deg": theta_deg,
                    "T1_s": pt["b0_results"][B0]["T1"], "T2_s": pt["b0_results"][B0]["T2"],
                    "fit_R2": pt["r2"], "tcf0": pt["tcf0"],
                })

        n_ok += 1

    summary = pd.DataFrame(summary_rows)
    detailed = pd.DataFrame(detailed_rows)
    per_theta_df = pd.DataFrame(per_theta_rows)
    missing = pd.DataFrame(missing_rows)

    summary.to_csv(args.out_summary, index=False)
    detailed.to_csv(args.out_detailed, index=False)
    per_theta_df.to_csv(args.out_per_theta, index=False)
    missing.to_csv(args.out_missing, index=False)

    print(f"Systems with complete powder-averaged results: {n_ok}")
    print(f"Systems with something missing (old and/or new): {n_partial}")
    print(f"Wrote {args.out_summary} ({len(summary)} rows)")
    print(f"Wrote {args.out_detailed} ({len(detailed)} rows)")
    print(f"Wrote {args.out_per_theta} ({len(per_theta_df)} rows)")
    print(f"Wrote {args.out_missing} ({len(missing)} rows)")


if __name__ == "__main__":
    main()
