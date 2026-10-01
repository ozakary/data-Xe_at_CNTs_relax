#!/usr/bin/env python3
"""
Symbolic verification of the reduced Wigner rotation matrix elements and
the resulting h_lab,0(t), h_lab,1(t) expressions used in ESI Section I.4
("Relaxation at a general SWCNT-to-B0 tilt angle").

This script does not depend on any of this project's own code, it checks
Eq. si_wigner_matrix and Eq. si_h_lab_0 against an independent, third-
party symbolic implementation of angular momentum theory (SymPy's own
Wigner-d function, sympy.physics.quantum.spin.Rotation.d), not against
this project's own prior derivation. Run it directly to reproduce every
symbolic result quoted in that subsection, nothing here is a canned
printout, every expression below is derived fresh by SymPy each run.

Three things are checked, matching the three claims made in the text:

  1. The reduced Wigner d-matrix elements, rows m=0 and m=1 (the only two
     rows that ever enter the spin-1/2 relaxation Hamiltonian, at any
     theta, by the A^(+/-2)=0 selection rule already used at theta=0),
     match Eq. si_wigner_matrix exactly.
  2. Both rows reduce to the identity at theta=0 (d2_00(0)=1, d2_11(0)=1,
     every other entry in these two rows =0), the algebraic statement
     behind "Eq. si_wigner reduces exactly to Eq. si_h0h1 at theta=0".
  3. Substituting these matrix elements, together with the real-tensor
     phase relation h_tube,(-m) = (-1)^m h_tube,(m)*, into Eq. si_wigner
     gives exactly Eq. si_h_lab_0 for h_lab,0(t) (a real sum of three
     terms), and shows explicitly why h_lab,1(t) does NOT collapse the
     same way (the h_tube,(1) and h_tube,(1)* coefficients differ there).

Usage
-----
    python verify_wigner_matrix.py
"""

import sympy as sp
from sympy.physics.quantum.spin import Rotation

theta = sp.symbols("theta", real=True)


def get_reduced_wigner_row(m: int):
    """Row m of the rank-2 reduced Wigner d-matrix, k = -2..2, from
    SymPy's own independent implementation."""
    return [sp.simplify(Rotation.d(2, m, k, theta).doit()) for k in range(-2, 3)]


def main():
    print("=" * 70)
    print("STEP 1: reduced Wigner d-matrix, rows m=0 and m=1, k=-2..2")
    print("=" * 70)
    row0 = get_reduced_wigner_row(0)
    row1 = get_reduced_wigner_row(1)
    for label, row in [("d^2_{0,k}", row0), ("d^2_{1,k}", row1)]:
        print(f"\n{label}(theta), k = -2,-1,0,1,2:")
        for k, val in zip(range(-2, 3), row):
            print(f"  k={k:2d}: {val}")
        print(f"  LaTeX: {sp.latex(row)}")

    # Eq. si_wigner_matrix, as published, for direct comparison
    published_row0 = [
        sp.sqrt(6) / 4 * sp.sin(theta) ** 2,
        -sp.sqrt(6) / 4 * sp.sin(2 * theta),
        sp.Rational(1, 2) * (3 * sp.cos(theta) ** 2 - 1),
        sp.sqrt(6) / 4 * sp.sin(2 * theta),
        sp.sqrt(6) / 4 * sp.sin(theta) ** 2,
    ]
    published_row1 = [
        sp.Rational(1, 2) * (sp.cos(theta) - 1) * sp.sin(theta),
        sp.Rational(1, 2) * (sp.cos(theta) - sp.cos(2 * theta)),
        -sp.sqrt(6) / 4 * sp.sin(2 * theta),
        sp.Rational(1, 2) * (sp.cos(theta) + sp.cos(2 * theta)),
        sp.Rational(1, 2) * (sp.cos(theta) + 1) * sp.sin(theta),
    ]

    print("\n" + "=" * 70)
    print("Cross-check against Eq. si_wigner_matrix as published:")
    print("=" * 70)
    ok = True
    for label, sympy_row, published in [("m=0", row0, published_row0), ("m=1", row1, published_row1)]:
        for k, (a, b) in zip(range(-2, 3), zip(sympy_row, published)):
            match = sp.simplify(a - b) == 0
            ok &= match
            print(f"  {label}, k={k:2d}: match = {match}")
    print(f"\n  ALL ELEMENTS MATCH: {ok}")
    assert ok, "Published Eq. si_wigner_matrix does not match SymPy's independent Wigner-d implementation"

    print("\n" + "=" * 70)
    print("STEP 2: theta=0 reduction")
    print("=" * 70)
    row0_at_0 = [v.subs(theta, 0) for v in row0]
    row1_at_0 = [v.subs(theta, 0) for v in row1]
    print(f"  d^2_{{0,k}}(0) = {row0_at_0}   (expect [0,0,1,0,0])")
    print(f"  d^2_{{1,k}}(0) = {row1_at_0}   (expect [0,0,0,1,0])")
    assert row0_at_0 == [0, 0, 1, 0, 0]
    assert row1_at_0 == [0, 0, 0, 1, 0]
    print("  Confirmed: h_lab,0(0)=h_tube,0(t), h_lab,1(0)=h_tube,1(t), Eq. si_wigner -> Eq. si_h0h1 exactly.")

    print("\n" + "=" * 70)
    print("STEP 3: explicit h_lab,0(t) and h_lab,1(t), via the phase relation")
    print("h_tube,(-m) = (-1)^m h_tube,(m)*")
    print("=" * 70)
    h0, h1, h1c, h2, h2c = sp.symbols("h_tube0 h_tube1 h_tube1c h_tube2 h_tube2c")
    # k order matches the rows above: k = -2,-1,0,1,2
    tube_terms = [h2c, -h1c, h0, h1, h2]  # h_{-1}=-h1*, h_{-2}=+h2*

    h_lab_0 = sp.expand(sum(c * t for c, t in zip(row0, tube_terms)))
    h_lab_1 = sp.expand(sum(c * t for c, t in zip(row1, tube_terms)))

    print("\nh_lab,0(t), collected by tube-frame term:")
    for sym in (h0, h1, h1c, h2, h2c):
        print(f"  coefficient of {sym}: {sp.simplify(h_lab_0.coeff(sym))}")
    h1_match_0 = sp.simplify(h_lab_0.coeff(h1) - h_lab_0.coeff(h1c)) == 0
    h2_match_0 = sp.simplify(h_lab_0.coeff(h2) - h_lab_0.coeff(h2c)) == 0
    print(f"  h1 and h1c coefficients equal (-> collapses to Re[h_tube,1]): {h1_match_0}")
    print(f"  h2 and h2c coefficients equal (-> collapses to Re[h_tube,2]): {h2_match_0}")
    assert h1_match_0 and h2_match_0, "h_lab,0(t) does not collapse to the claimed real 3-term form"

    print("\nh_lab,1(t), collected by tube-frame term:")
    for sym in (h0, h1, h1c, h2, h2c):
        print(f"  coefficient of {sym}: {sp.simplify(h_lab_1.coeff(sym))}")
    h1_match_1 = sp.simplify(h_lab_1.coeff(h1) - h_lab_1.coeff(h1c)) == 0
    h2_match_1 = sp.simplify(h_lab_1.coeff(h2) - h_lab_1.coeff(h2c)) == 0
    print(f"  h1 and h1c coefficients equal: {h1_match_1}  (expect False, does NOT collapse this way)")
    print(f"  h2 and h2c coefficients equal: {h2_match_1}  (expect False, does NOT collapse this way)")
    assert not h1_match_1, "h_lab,1(t) unexpectedly collapses the same way h_lab,0(t) does, text needs revising"

    print("\n" + "=" * 70)
    print("Final published form, Eq. si_h_lab_0 (LaTeX):")
    print("=" * 70)
    A, B, C = [sp.simplify(2 * h_lab_0.coeff(s)) for s in (h0, h1, h2)]
    print(f"  h_lab,0(t) = ({sp.latex(sp.Rational(1,1)*A/2)}) h_tube,0(t)"
          f" + ({sp.latex(B)}) Re[h_tube,(1)(t)]"
          f" + ({sp.latex(C)}) Re[h_tube,(2)(t)]")


if __name__ == "__main__":
    main()
