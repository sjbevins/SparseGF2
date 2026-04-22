"""
Extract 4x4 GF(2) symplectic matrices from Stim two-qubit Clifford tableaux.

Given a stim.Tableau for a 2-qubit Clifford gate, the 4x4 symplectic matrix S
describes how the gate transforms the basis (X_0, X_1, Z_0, Z_1):

    (x'_0, x'_1, z'_0, z'_1) = (x_0, x_1, z_0, z_1) * S   (mod 2)

Row i of S is the output of applying the gate to the i-th basis element.
"""

import numpy as np


def symplectic_from_stim_tableau(tab) -> np.ndarray:
    """Extract a 4x4 GF(2) symplectic matrix from a Stim 2-qubit Tableau.

    Parameters
    ----------
    tab : stim.Tableau
        A 2-qubit Clifford tableau from Stim.

    Returns
    -------
    S : ndarray of uint8, shape (4, 4)
        The GF(2) symplectic matrix. Row order: X_0, X_1, Z_0, Z_1.
        Column order: X'_0, X'_1, Z'_0, Z'_1.
    """
    S = np.zeros((4, 4), dtype=np.uint8)

    # Basis inputs: X_0, X_1, Z_0, Z_1
    for inp_idx, output_ps in enumerate([
        tab.x_output(0), tab.x_output(1),
        tab.z_output(0), tab.z_output(1),
    ]):
        for q in range(2):
            pauli = output_ps[q]  # 0=I, 1=X, 2=Y, 3=Z
            S[inp_idx, q]     = 1 if pauli in (1, 2) else 0  # X or Y
            S[inp_idx, q + 2] = 1 if pauli in (2, 3) else 0  # Y or Z

    return S
