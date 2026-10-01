# Preprocessing and TCF Construction

Builds the run manifest, ingests raw MLMD trajectories into a per-system array, and constructs the shielding time correlation function (TCF) at θ=0 (SWCNT axis parallel to B<sub>0</sub>).

## Scripts

### [`build_run_manifest.py`](./build_run_manifest.py)
Scans the directory tree of generated systems and builds the manifest CSV used by every downstream script in this repository (`run_manifest.csv`), assigning each system to its case (temperature, diameter, or loading series) and subcase (LL/HL or chirality).

Requires a hand-authored `cnt_systems_info.csv` listing, one row per system, at minimum: `system_name`, `chirality`, `n_Xe`, `case`, `subcase`, `variable_value` (the swept quantity's value for that system, e.g. temperature in K, diameter in Å, or loading in %).

```bash
python build_run_manifest.py --root .. --systems-info ./cnt_systems_info.csv --out ./run_manifest.csv
```

### [`step0_verify_xyz_id_vs_csv_index.py`](./step0_verify_xyz_id_vs_csv_index.py)
Sanity check confirming that atom indices in the raw trajectory files match the indexing assumed by the NMR-ML shielding predictions, before any further processing.

### [`step1_ingest_and_track.py`](./step1_ingest_and_track.py)
Ingests one system's raw MLMD trajectory and NMR-ML shielding predictions, tracks each Xe atom across frames, identifies and corrects outlier shielding values via Tukey's IQR rule, and assembles everything into one `_assembled.npz` file consumed by every later step.

```bash
python step1_ingest_and_track.py \
    --data-dir ./cnt_10_0_u23_Xe4_10ns \
    --system-name cnt_10_0_u23_Xe4_10ns \
    --dt-fs 100.0 \
    --n-workers 8 \
    --out ./cnt_10_0_u23_Xe4_10ns_assembled.npz
```

### [`step2_build_tcf.py`](./step2_build_tcf.py)
Constructs h₀(t), h₁(t) at θ=0 from the assembled tensor and builds the per-atom and pooled shielding TCF via FFT.

```bash
python step2_build_tcf.py \
    --assembled ./cnt_10_0_u23_Xe4_10ns_assembled.npz \
    --out ./cnt_10_0_u23_Xe4_10ns_tcf.npz \
    --plot ./cnt_10_0_u23_Xe4_10ns_tcf.png \
    --plot-max-lag-ps 20
```

### [`step2b_tcf_diagnostics.py`](./step2b_tcf_diagnostics.py)
Diagnostic check on the θ=0 TCF, confirming it decays to a stable noise floor within the trajectory length, before fitting.

```bash
python step2b_tcf_diagnostics.py \
    --tcf ./cnt_10_0_u23_Xe4_10ns_tcf.npz \
    --out-prefix ./cnt_10_0_u23_Xe4_10ns_tcf_diag \
    --long-lag-ps 2000 \
    --noise-floor-frac 0.1
```

Run in order ([`step0`](./step0_verify_xyz_id_vs_csv_index.py) --> [`step1`](./step1_ingest_and_track.py) --> [`step2`](./step2_build_tcf.py) --> [`step2b`](./step2b_tcf_diagnostics.py)) once per system, after [`build_run_manifest.py`](./build_run_manifest.py) has been run once for the whole dataset.
