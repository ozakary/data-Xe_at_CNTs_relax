# Wigner Rotation Matrix Verification

### [`verify_wigner_matrix.py`](./verify_wigner_matrix.py)
Standalone symbolic check (SymPy) of the reduced Wigner d-matrix elements used throughout [`powder_averaged_relaxation/`](../powder_averaged_relaxation/), against an independent implementation (`sympy.physics.quantum.spin.Rotation.d`), not the derivation of this project. Confirms the published matrix elements, the θ=0 reduction to the original single-orientation treatment, and the explicit h<sub>lab,0</sub>(t), h<sub>lab,1</sub>(t) expansions used in the manuscript and SI.

Has no dependency on any other script or data file in this repository.

```bash
python verify_wigner_matrix.py
```
