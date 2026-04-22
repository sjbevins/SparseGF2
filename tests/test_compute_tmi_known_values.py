"""Ground-truth tests for compute_tmi against known closed-form values.

References:
  Hosur, Qi, Roberts, Yoshida 2016, arXiv:1511.04021, Eq. 24.
"""
from __future__ import annotations

import pytest

from sparsegf2 import SparseGF2


@pytest.mark.parametrize("n", [6, 9, 12])
def test_bell_pair_tmi_is_volume_law(n):
    """For Bell pairs (maximally entangled system+reference), each system
    subregion A has S(A) = |A|. The TMI formula evaluates to a negative
    volume-law value: I_3 = |A|+|B|+|C|-|AB|-|AC|-|BC|+|ABC| = 0 when the
    three parts tile the full system and every entropy is |region|.

    Let sizes be a, b, c with a+b+c = n. Then:
      I_3 = a + b + c - (a+b) - (a+c) - (b+c) + (a+b+c)
          = (a+b+c) - 2(a+b+c) + (a+b+c)
          = 0.
    So TMI is exactly 0 for the initial Bell-pair state.
    """
    sim = SparseGF2(n)
    tmi = sim.compute_tmi()
    assert tmi == 0, f"Bell-pair TMI should be 0, got {tmi}"


def test_tmi_after_full_measurement_is_zero():
    """After every system qubit is measured in Z, the system decouples
    from reference and S(A) = 0 for every A. TMI = 0.
    """
    n = 6
    sim = SparseGF2(n)
    for q in range(n):
        sim.apply_measurement_z(q)
    assert sim.compute_k() == 0
    assert sim.compute_tmi() == 0
