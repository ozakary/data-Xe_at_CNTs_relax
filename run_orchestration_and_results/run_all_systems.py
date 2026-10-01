#!/usr/bin/env python3
"""
Run the full pipeline for every system in the manifest built by
build_run_manifest.py, now including the powder-averaged tilt-angle
methodology alongside the original theta=0-only steps.

Steps run per system, in order:
  step1   - ingest and track (unchanged)
  step2   - build h0(t), h1(t) TCF at theta=0 (unchanged, kept for direct
            comparison against the powder-averaged result)
  step2b  - TCF diagnostics on the theta=0 TCF (unchanged, diagnostic only,
            never blocks the pipeline)
  step3   - theta=0 spectral density and relaxation (unchanged, kept as the
            "old" reference point estimate)
  step4_old   - per-track uncertainty at theta=0 (unchanged, kept as the
            "old" reference uncertainty)
  step2f  - NEW: powder-averaged relaxation (pooled TCF, 10-point angle
            grid locked after step2e's convergence test, sin(theta)-weighted
            average on rates). This is now the primary, reported point
            estimate, replacing step3's role.
  step4_new   - NEW: per-track powder-averaged uncertainty (each track fit
            at every angle, powder-averaged per track, then mean/SEM across
            tracks). This is now the primary, reported uncertainty.

The old and new steps are independent of each other (step2f and step4_new
both read directly from step1's assembled.npz, not from step2's tcf.npz),
so either half can be rerun or skipped without touching the other. Kept
side by side deliberately, for a direct theta=0-vs-powder-averaged
comparison per system once this reaches the ESI, not because both are
needed going forward.

Resumable exactly as before: each step is skipped if its output file
already exists, unless --force is given. Per-system logs go to
<results-dir>/<run_id>/log_stepN.txt.

Usage
-----
    python run_all_systems.py \\
        --manifest ./run_manifest.csv \\
        --scripts-dir . \\
        --results-dir ./results \\
        --fit-max-lag-ps 3000 \\
        --n-angles 10 \\
        --b0-fields 9.4 14.1 \\
        --n-workers 8

    # skip the old theta=0-only steps entirely, new methodology only:
    python run_all_systems.py ... --skip-old-pipeline

    # restrict to one case while testing, e.g. just the temperature sweep:
    python run_all_systems.py ... --only-case 1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def run_step(name, cmd, log_path: Path, skip_if_exists: Path | None, force: bool):
    if skip_if_exists is not None and skip_if_exists.exists() and not force:
        print(f"    [{name}] skip (already exists: {skip_if_exists.name})")
        return True
    t0 = time.time()
    with open(log_path, "w") as f:
        f.write("$ " + " ".join(str(c) for c in cmd) + "\n\n")
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    ok = result.returncode == 0
    status = "ok" if ok else f"FAILED (see {log_path})"
    print(f"    [{name}] {status}  ({dt:.1f}s)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--scripts-dir", required=True, type=Path, help="directory containing all stepN .py files")
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--dt-fs", type=float, default=100.0)
    ap.add_argument("--fit-max-lag-ps", type=float, default=3000.0)
    ap.add_argument("--n-angles", type=int, default=10,
                     help="powder-average angle grid size, locked to 10 after step2e's convergence test")
    ap.add_argument("--b0-fields", type=float, nargs="+", default=[9.4, 14.1])
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--only-case", type=int, default=None, choices=[1, 2, 3])
    ap.add_argument("--force", action="store_true", help="rerun steps even if output already exists")
    ap.add_argument("--skip-old-pipeline", action="store_true",
                     help="run only step1 + the new powder-averaged steps (2f, new 4), skip steps 2/2b/3/old-4")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.only_case is not None:
        manifest = manifest[manifest["case"] == args.only_case]

    args.results_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    scripts = args.scripts_dir

    n_ok, n_fail = 0, 0
    failed_systems = []

    for i, row in manifest.iterrows():
        name = row["run_id"]
        data_dir = Path(row["data_dir_full"])
        out_dir = args.results_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{i+1}/{len(manifest)}] {name}  (case {row['case']}, {row['data_dir']})")

        if not data_dir.exists():
            print(f"    SKIPPING ENTIRELY, data dir does not exist: {data_dir}")
            n_fail += 1
            failed_systems.append((name, "data_dir missing"))
            continue

        assembled = out_dir / f"{name}_assembled.npz"
        ok1 = run_step("step1", [
            py, str(scripts / "step1_ingest_and_track.py"),
            "--data-dir", str(data_dir), "--system-name", name,
            "--dt-fs", str(args.dt_fs), "--n-workers", str(args.n_workers),
            "--out", str(assembled),
        ], out_dir / "log_step1.txt", assembled, args.force)
        if not ok1:
            n_fail += 1; failed_systems.append((name, "step1")); continue

        if not args.skip_old_pipeline:
            tcf = out_dir / f"{name}_tcf.npz"
            tcf_plot = out_dir / f"{name}_tcf.png"
            ok2 = run_step("step2 (theta=0)", [
                py, str(scripts / "step2_build_tcf.py"),
                "--assembled", str(assembled), "--out", str(tcf), "--plot", str(tcf_plot),
            ], out_dir / "log_step2.txt", tcf, args.force)

            if ok2:
                diag_prefix = out_dir / f"{name}_tcf_diag"
                run_step("step2b (diagnostics)", [
                    py, str(scripts / "step2b_tcf_diagnostics.py"),
                    "--tcf", str(tcf), "--out-prefix", str(diag_prefix),
                    "--long-lag-ps", str(args.fit_max_lag_ps),
                ], out_dir / "log_step2b.txt", Path(f"{diag_prefix}.png"), args.force)
                # diagnostic only, never blocks the pipeline

                relax = out_dir / f"{name}_relaxation.npz"
                fit_plot = out_dir / f"{name}_fit.png"
                run_step("step3 (theta=0)", [
                    py, str(scripts / "step3_spectral_density_and_relaxation.py"),
                    "--tcf", str(tcf), "--fit-max-lag-ps", str(args.fit_max_lag_ps),
                    "--b0-fields", *[str(b) for b in args.b0_fields],
                    "--out", str(relax), "--plot", str(fit_plot),
                ], out_dir / "log_step3.txt", relax, args.force)

                relax_pt = out_dir / f"{name}_relaxation_per_track.npz"
                run_step("step4_old (theta=0)", [
                    py, str(scripts / "step4_per_track_uncertainty.py"),
                    "--tcf", str(tcf), "--fit-max-lag-ps", str(args.fit_max_lag_ps),
                    "--b0-fields", *[str(b) for b in args.b0_fields],
                    "--out", str(relax_pt),
                ], out_dir / "log_step4_old.txt", relax_pt, args.force)
            else:
                print("    (step2 failed, skipping 2b/3/old-4 for this system, new powder-averaged "
                      "steps below are independent and will still run)")

        # New powder-averaged methodology, reads assembled.npz directly,
        # independent of the old steps above.
        powder_relax = out_dir / f"{name}_powder_relaxation.npz"
        powder_plot = out_dir / f"{name}_powder_relaxation.png"
        ok2f = run_step("step2f (powder avg)", [
            py, str(scripts / "step2f_powder_averaged_relaxation.py"),
            "--assembled", str(assembled), "--n-angles", str(args.n_angles),
            "--fit-max-lag-ps", str(args.fit_max_lag_ps),
            "--b0-fields", *[str(b) for b in args.b0_fields],
            "--out", str(powder_relax), "--plot", str(powder_plot),
        ], out_dir / "log_step2f.txt", powder_relax, args.force)
        if not ok2f:
            n_fail += 1; failed_systems.append((name, "step2f")); continue

        powder_relax_pt = out_dir / f"{name}_powder_relaxation_per_track.npz"
        ok4new = run_step("step4_new (powder avg per-track)", [
            py, str(scripts / "step4_powder_averaged_per_track_uncertainty.py"),
            "--assembled", str(assembled), "--n-angles", str(args.n_angles),
            "--fit-max-lag-ps", str(args.fit_max_lag_ps),
            "--b0-fields", *[str(b) for b in args.b0_fields],
            "--out", str(powder_relax_pt),
        ], out_dir / "log_step4_new.txt", powder_relax_pt, args.force)
        if not ok4new:
            n_fail += 1; failed_systems.append((name, "step4_new")); continue

        n_ok += 1

    print(f"\nDone: {n_ok} ok, {n_fail} failed out of {len(manifest)}")
    if failed_systems:
        print("Failed:")
        for name, stage in failed_systems:
            print(f"  {name}: {stage}")


if __name__ == "__main__":
    main()
