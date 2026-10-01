# Batch Orchestration and Results Collection

Runs the full pipeline across all 38 systems and collects the per-system `.npz` outputs into flat summary CSVs for the figures and tables.

## Scripts

### [`run_all_systems.py`](./run_all_systems.py)
Runs the preprocessing, θ=0 reference, and powder-averaged pipelines for every system in the manifest, in parallel.

```bash
# From scratch
python run_all_systems.py \
    --manifest ./run_manifest.csv \
    --scripts-dir . \
    --results-dir ./results \
    --fit-max-lag-ps 3000 \
    --n-angles 10 \
    --b0-fields 9.4 14.1 \
    --n-workers 8 \
    --force

# Re-run, reusing any already-computed per-system results
python run_all_systems.py \
    --manifest ./run_manifest.csv \
    --scripts-dir . \
    --results-dir ./results \
    --fit-max-lag-ps 3000 \
    --n-angles 10 \
    --b0-fields 9.4 14.1 \
    --n-workers 8
```

### [`collect_results.py`](./collect_results.py)
Collects every system's `.npz` outputs into four flat CSVs: a summary (powder-averaged and θ=0 T<sub>1</sub>/T<sub>2</sub> side by side), a detailed per-angle, per-atom breakdown, a per-theta table (input to the angular figures), and a list of any systems missing expected output.

```bash
python collect_results.py \
    --manifest ./run_manifest.csv \
    --results-dir ./results \
    --out-summary ./results_summary.csv \
    --out-detailed ./results_detailed.csv \
    --out-per-theta ./results_per_theta.csv \
    --out-missing ./results_missing.csv
```

Run [`run_all_systems.py`](./run_all_systems.py) after [`preprocessing_and_tcf/build_run_manifest.py`](../preprocessing_and_tcf/build_run_manifest.py) has produced the manifest, and [`collect_results.py`](./collect_results.py) after [`run_all_systems.py`](./run_all_systems.py) completes. [`step6_check_cutoff_robustness_powder.py`](../validation_cutoff_and_angle_grid/step6_check_cutoff_robustness_powder.py) (`validation_cutoff_and_angle_grid/`) also depends on this directory's output, run after this step.
