# Powder-Averaged Uncertainty Estimation

### [`step4_powder_averaged_per_track_uncertainty.py`](./step4_powder_averaged_per_track_uncertainty.py)
Fits each per-atom TCF independently at every angle in the 10-point grid, powder-averages T<sub>1</sub>(θ), T<sub>2</sub>(θ) sequence for each Xe atom, and reports the standard error of the mean of these per-atom powder averages across atoms as the uncertainty on the reported T<sub>1</sub>, T<sub>2</sub>.

This is a distinct calculation from [`powder_averaged_relaxation/step2f_powder_averaged_relaxation.py`](../powder_averaged_relaxation/step2f_powder_averaged_relaxation.py), which pools all per-atom TCFs before fitting (the lower-noise point estimate). This script never pools, fitting every per-atom TCF separately (the uncertainty estimate).

```bash
python step4_powder_averaged_per_track_uncertainty.py \
    --assembled ./cnt_10_0_u23_Xe4_10ns_assembled.npz \
    --n-angles 10 \
    --fit-max-lag-ps 3000 \
    --b0-fields 9.4 14.1 \
    --out ./cnt_10_0_u23_Xe4_10ns_powder_relaxation_per_track.npz
```

Run after [`preprocessing_and_tcf/`](../preprocessing_and_tcf/), alongside [`powder_averaged_relaxation/`](../powder_averaged_relaxation/).
