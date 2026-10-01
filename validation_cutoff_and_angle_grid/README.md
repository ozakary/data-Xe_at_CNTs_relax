# Fit-Window and Robustness Validation

Confirms that the 3000 ps TCF fit window is long enough for both the θ=0-only and powder-averaged methodologies, across all 38 systems and both fields.

## Scripts

### [`step5_cutoff_sensitivity.py`](./step5_cutoff_sensitivity.py)
Per-system check of the θ=0 fit window: refits a single per-system TCF over a range of cutoffs and tracks how T<sub>1</sub>, T<sub>2</sub> change as the window grows.

```bash
python step5_cutoff_sensitivity.py \
    --tcf ./cnt_10_0_u23_Xe4_10ns_tcf.npz \
    --cutoffs-ps 300 500 750 1000 1500 2000 3000 \
    --b0-fields 9.4 14.1 \
    --out ./cnt_10_0_u23_Xe4_10ns_cutoff_sensitivity.npz \
    --plot ./cnt_10_0_u23_Xe4_10ns_cutoff_sensitivity.png
```

### [`step2g_cutoff_robustness_powder.py`](./step2g_cutoff_robustness_powder.py)
The same check, per system, for the powder-averaged methodology: refits every angle in the grid at each cutoff and powder-averages at each, tracking convergence of the powder-averaged T<sub>1</sub>, T<sub>2</sub>.

```bash
python step2g_cutoff_robustness_powder.py \
    --assembled ./cnt_10_0_u23_Xe4_10ns_assembled.npz \
    --n-angles 10 \
    --cutoffs-ps 20 50 100 200 300 500 750 1000 1500 2000 3000 \
    --reference-cutoff-ps 3000 \
    --b0-fields 9.4 14.1 \
    --out ./cnt_10_0_u23_Xe4_10ns_powder_cutoff_robustness.npz \
    --plot ./cnt_10_0_u23_Xe4_10ns_powder_cutoff_robustness.png
```

### [`step6_check_cutoff_robustness_powder.py`](./step6_check_cutoff_robustness_powder.py)
Runs the [`step2g`](./step2g_cutoff_robustness_powder.py) check across all 38 systems at once and reports what fraction fall within tolerance (10%) of the 3000 ps reference at each candidate cutoff, the basis for Figures S1-S6.

```bash
python step6_check_cutoff_robustness_powder.py \
    --manifest ./run_manifest.csv \
    --results-dir ./results \
    --n-angles 10 \
    --cutoffs-ps 20 50 100 200 300 500 750 1000 1500 2000 3000 \
    --reference-cutoff-ps 3000 \
    --tolerance-pct 10 \
    --b0-fields 9.4 14.1 \
    --out-detailed ./powder_cutoff_robustness_detailed.csv \
    --out-summary ./powder_cutoff_robustness_summary.csv
```

[`step6`](./step6_check_cutoff_robustness_powder.py) must be run after [`run_orchestration_and_results/run_all_systems.py`](../run_orchestration_and_results/run_all_systems.py) has generated the per-system results directory for all 38 systems.
