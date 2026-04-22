"""
Physics-picture initialization for graph-defined circuits.

Two pictures are supported:

- ``purification``: 2n qubits (n system + n reference), initialized as n
  Bell pairs with Cliffords and measurements acting only on the system
  qubits. Reference: Gullans & Huse, "Dynamical Purification Phase
  Transition Induced by Quantum Measurements", Phys. Rev. X 10, 041020
  (2020), arXiv:1905.05195. Returned simulator: :class:`SparseGF2`.

- ``single_ref``: n+1 qubits total (n system + 1 reference), initialized
  in ``|0...0>`` and then entangled via ``H(0); CNOT(0 -> n)`` so that
  system qubit 0 forms a single Bell pair with reference qubit n; the
  other system qubits (1..n-1) start in ``|0>``. Cliffords and
  measurements act only on qubits 0..n-1; qubit n is an untouched probe.
  After the circuit, the reduced entropy S(qubit n) diagnoses MIPT: at
  low measurement rate it is ~1 (volume-law side, reference stays
  entangled with the scrambled system), at high rate it is 0 (area-law
  side, measurement has fully collapsed the Bell-pair correlation).
  Returned simulator: :class:`StabilizerTableau` (dense, supports
  arbitrary qubit count).
"""
from __future__ import annotations

from typing import Tuple, Union

from sparsegf2 import SparseGF2, StabilizerTableau


PICTURES: Tuple[str, ...] = ("purification", "single_ref")


def init_picture(
    picture: str,
    n: int,
    *,
    use_min_weight_pivot: bool = True,
    check_inputs: bool = False,
    hybrid_mode: bool = True,
) -> Union[SparseGF2, StabilizerTableau]:
    """Initialize a stabilizer simulator in the named physics picture.

    Parameters
    ----------
    picture : str
        One of :data:`PICTURES`.
    n : int
        Number of system qubits.
    use_min_weight_pivot, check_inputs, hybrid_mode
        Forwarded to :class:`SparseGF2` when ``picture == "purification"``;
        ignored for ``single_ref``.

    Returns
    -------
    SparseGF2 or StabilizerTableau
        A simulator whose state is the initial state of the requested picture.

        - ``purification`` -> :class:`SparseGF2` with 2n qubits, initialized
          to n Bell pairs between system qubits 0..n-1 and reference qubits
          n..2n-1.
        - ``single_ref`` -> :class:`StabilizerTableau` with n+1 qubits,
          initialized to a product state containing one Bell pair between
          qubit 0 and qubit n (the reference); qubits 1..n-1 start in |0>.
    """
    if picture == "purification":
        return SparseGF2(
            n,
            use_min_weight_pivot=use_min_weight_pivot,
            check_inputs=check_inputs,
            hybrid_mode=hybrid_mode,
        )
    if picture == "single_ref":
        tab = StabilizerTableau.from_zero_state(n + 1)
        tab.h(0)
        tab.cnot(0, n)
        return tab
    raise ValueError(
        f"picture must be one of {PICTURES}; got {picture!r}."
    )


__all__ = ["PICTURES", "init_picture"]
