# Manuscript and SI Table Generation

### [`print_t1_t2_summary_table.py`](./print_t1_t2_summary_table.py)
Prints LaTeX rows for Tables S7-S9, powder-averaged T<sub>1</sub>/T<sub>2</sub> ± SEM and the angular spread δT<sub>1</sub>/δT<sub>2</sub>, one block per series.

```bash
python print_t1_t2_summary_table.py --manifest ../run_manifest.csv --results-dir ../results --b0-fields 9.4 14.1
```

### [`print_tcf_fit_params_table.py`](./print_tcf_fit_params_table.py)
Prints LaTeX rows for Tables S1-S6, TCF fit parameters (A<sub>1</sub>, τ<sub>1</sub>, A<sub>2</sub>, τ<sub>2</sub>, R<sup>2</sup>, ⟨h<sub>0</sub>⟩) at θ=0, 40, 90°, one block per series.

```bash
python print_tcf_fit_params_table.py --manifest ../run_manifest.csv --results-dir ../results --theta-deg 0 40 90
```

Both scripts read the manifest from [`run_orchestration_and_results/`](../run_orchestration_and_results/) and the per-system results directory, run after that step completes.
