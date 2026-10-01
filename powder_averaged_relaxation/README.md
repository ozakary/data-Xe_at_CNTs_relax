# Powder-Averaged Relaxation at a General Tilt Angle

The primary methodology of this work. Extends the θ=0 treatment to a general SWCNT-to-B<sub>0</sub> tilt angle θ via Wigner rotation, and powder-averages the resulting T<sub>1</sub>(θ), T<sub>2</sub>(θ) over θ to obtain the reported relaxation times.

## Scripts

### [`step2c_build_h2_tcf.py`](./step2c_build_h2_tcf.py)
Constructs the tube-frame h<sub>2</sub>(t) component (absent from the θ=0 Hamiltonian, required at every θ≠0) and its TCF, and checks its cross-correlation with h<sub>0</sub>(t), h<sub>1</sub>(t) to confirm the expected azimuthal symmetry.

```bash
python step2c_build_h2_tcf.py \
    --assembled ./cnt_10_0_u23_Xe4_10ns_assembled.npz \
    --tcf ./cnt_10_0_u23_Xe4_10ns_tcf.npz \
    --out ./cnt_10_0_u23_Xe4_10ns_h2_tcf.npz \
    --plot ./cnt_10_0_u23_Xe4_10ns_h2_tcf.png
```

### [`step2d_tilt_angle_relaxation.py`](./step2d_tilt_angle_relaxation.py)
Exploratory/diagnostic script computing T<sub>1</sub>(θ), T<sub>2</sub>(θ) at a hand-picked set of angles (including the magic angle, 54.7°), used during development to inspect the angular dependence before locking the production grid.

```bash
python step2d_tilt_angle_relaxation.py \
    --assembled ./cnt_10_0_u23_Xe4_10ns_assembled.npz \
    --theta-deg 0 15 30 45 54.7 60 75 90 \
    --fit-max-lag-ps 3000 \
    --b0-fields 9.4 14.1 \
    --out ./cnt_10_0_u23_Xe4_10ns_tilt_relaxation.npz
```

### [`step2e_angle_grid_convergence.py`](./step2e_angle_grid_convergence.py)
Validates the production 10-point angle grid against a much finer, 46-point reference grid, on both a narrow and a wide SWCNT, confirming the locked grid converges to within 0.5%.

```bash
python step2e_angle_grid_convergence.py \
    --assembled ./cnt_10_0_u23_Xe4_10ns_assembled.npz \
    --reference-n-angles 46 \
    --candidate-grids "0,90" "0,45,90" "0,30,60,90" "0,22.5,45,67.5,90" \
                      "0,15,30,45,60,75,90" "0,10,20,30,40,50,60,70,80,90" \
    --fit-max-lag-ps 3000 \
    --b0-fields 9.4 14.1 \
    --out ./cnt_10_0_u23_Xe4_10ns_angle_grid_convergence.npz \
    --plot ./cnt_10_0_u23_Xe4_10ns_angle_grid_convergence.png
```

### [`step2f_powder_averaged_relaxation.py`](./step2f_powder_averaged_relaxation.py)
The production script. Sweeps the locked 10-point angle grid (0° to 90°), builds the lab-frame h<sub>lab,0</sub>(t), h<sub>lab,1</sub>(t) at each angle via Wigner rotation, fits the pooled TCF, obtains T<sub>1</sub>(θ), T<sub>2</sub>(θ), and powder-averages the rates (sin θ weighted) to give the reported T<sub>1</sub>, T<sub>2</sub>, along with the orientational spread (T<sub>1</sub>/T<sub>2</sub> min/max across θ).

```bash
python step2f_powder_averaged_relaxation.py \
    --assembled ./cnt_10_0_u23_Xe4_10ns_assembled.npz \
    --n-angles 10 \
    --fit-max-lag-ps 3000 \
    --b0-fields 9.4 14.1 \
    --out ./cnt_10_0_u23_Xe4_10ns_powder_relaxation.npz \
    --plot ./cnt_10_0_u23_Xe4_10ns_powder_relaxation.png
```

Run after [`preprocessing_and_tcf/`](../preprocessing_and_tcf/). [`step2f_powder_averaged_relaxation.py`](./step2f_powder_averaged_relaxation.py) is the one script here whose output every later figure and table script depends on.
