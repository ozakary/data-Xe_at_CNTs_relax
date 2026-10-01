# Manuscript and SI Figure Generation

All figure scripts share [`figure_formatting_v2.py`](./figure_formatting_v2.py) for consistent fonts, colors, and fixed-frame layout, and read from [`run_orchestration_and_results/`](../run_orchestration_and_results/)'s output CSVs and the per-system results directory. Run after that step completes.

## Scripts

### [`plot_T1_angular_all_cases.py`](./plot_T1_angular_all_cases.py) / [`plot_T2_angular_all_cases.py`](./plot_T2_angular_all_cases.py)
T<sub>1</sub>(θ)/T<sub>2</sub>(θ) across the full angle grid, one SVG per (case, subcase), Figure S25 and its T<sub>2</sub> equivalent.

```bash
python plot_T1_angular_all_cases.py --per-theta-csv ../results_per_theta.csv --results-dir ../results --b0-fields 9.4 14.1
python plot_T2_angular_all_cases.py --per-theta-csv ../results_per_theta.csv --results-dir ../results --b0-fields 9.4 14.1
```

### [`plot_T1_all_cases.py`](./plot_T1_all_cases.py) / [`plot_T2_all_cases.py`](./plot_T2_all_cases.py)
Powder-averaged T<sub>1</sub>/T<sub>2</sub> versus the swept variable, one SVG per case, Figure 2's T<sub>1</sub>/T<sub>2</sub> panels.

```bash
python plot_T1_all_cases.py --manifest ../run_manifest.csv --results-dir ../results --b0-fields 9.4 14.1
python plot_T2_all_cases.py --manifest ../run_manifest.csv --results-dir ../results --b0-fields 9.4 14.1
```

### [`plot_spectral_density_all_cases.py`](./plot_spectral_density_all_cases.py) / [`plot_tcf_all_cases.py`](./plot_tcf_all_cases.py)
SDF and TCF, θ=0, one SVG per case, Figure 2's SDF/TCF panels.

```bash
python plot_spectral_density_all_cases.py --manifest ../run_manifest.csv --results-dir ../results --b0-fields 9.4 14.1
python plot_tcf_all_cases.py --manifest ../run_manifest.csv --results-dir ../results --x-left 0 1 --x-right 5 100
```

### [`plot_tcf_fit_diagnostics.py`](./plot_tcf_fit_diagnostics.py)
Per-system TCF-vs-fit diagnostic (data, mono-exp fit, bi-exp fit, residuals), at θ=0, 40, 90°, Figures S7-S24.

```bash
python plot_tcf_fit_diagnostics.py --manifest ../run_manifest.csv --results-dir ../results --theta-deg 0 40 90 --fit-max-lag-ps 3000 --xlim-max-ps 100 --inset-frac 0.2
```

### [`plot_cutoff_diagnostics.py`](./plot_cutoff_diagnostics.py)
Per-system fit-window sensitivity, powder-averaged T<sub>1</sub>/T<sub>2</sub> versus cutoff, Figures S1-S6.

```bash
python plot_cutoff_diagnostics.py --detailed ../powder_cutoff_robustness_detailed.csv --results-dir ../results --b0-field 9.4
```

### [`plot_figure3_gas_comparison.py`](./plot_figure3_gas_comparison.py)
Bar chart comparing this work's density-matched Xe@SWCNT T<sub>1</sub> against Hanni et al.'s experimental free-gas T<sub>1</sub>, Figure 3.

Reads from a hand-authored `figure3_data.csv`, one row per bar, columns: `label`, `density_amg`, `source` (`calc`/`expt`), `pair` (groups the two bars of one comparison), `T1_s`, `T1_sem_s`, `note`.

```bash
python plot_figure3_gas_comparison.py --csv ./figure3_data.csv --out ./figure3.svg
```
