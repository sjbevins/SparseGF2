"""Expurgation of low-depth random-circuit codes on the SparseGF2 tableau.

Implements the expurgation algorithm of Gullans, Krastanov, Huse,
Jiang, and Flammia, *Quantum Coding with Low-Depth Random Circuits*,
PRX **11**, 031066 (2021)
(`arXiv:2010.09775 <https://arxiv.org/abs/2010.09775>`_), Sec. VI,
natively on the phase-free tableau: repeatedly find error operators
the code cannot correct (zero syndrome, nontrivial logical action) and
remove them by projectively measuring them, spending one logical qubit
per measurement while distance and erasure recovery never decrease.

The package couples only to the public ``SparseGF2`` API
(:meth:`~sparsegf2.SparseGF2.measure_pauli`,
:meth:`~sparsegf2.SparseGF2.pauli_anticommuting_rows`) plus a role
array over destabilizer/stabilizer pairs, so it works with any tableau
regardless of how it was generated.

Layout:

* :mod:`~sparsegf2.expurgation.roles`: the ``(S, L, G)`` code view
  (:class:`StabilizerCode`) and the elementary measure-and-relabel move.
* :mod:`~sparsegf2.expurgation.erasure`: the erasure-decoding
  formalism: the uncorrectable-error matrix ``M``, its rank deficit
  ``r_M``, the exact recovery probability ``2**-r_M``, and candidate
  extraction via witness elimination.
* :mod:`~sparsegf2.expurgation.driver`: the published loop with
  mid-sequence re-validation and stopping criteria.
* :mod:`~sparsegf2.expurgation.encoding`: a random-encoding builder so
  the package runs standalone.
* :mod:`~sparsegf2.expurgation.purification`: extract the system code
  of a monitored purification run (system + reference qubits) into the
  presentation the machinery uses.

See ``src/sparsegf2/expurgation/README.md`` for a walkthrough.
"""

from sparsegf2.expurgation.driver import (
    ExpurgationConfig,
    ExpurgationResult,
    expurgate,
    mean_recovery,
)
from sparsegf2.expurgation.encoding import GEOMETRIES, random_encoding
from sparsegf2.expurgation.erasure import (
    Candidate,
    expurgation_candidates,
    recovery_probability,
    sample_erasure,
    uncorrectable_matrix,
    uncorrectable_rank,
)
from sparsegf2.expurgation.purification import from_purification
from sparsegf2.expurgation.roles import (
    ROLE_CHECK,
    ROLE_GAUGE,
    ROLE_LOGICAL,
    STRATEGIES,
    StabilizerCode,
)

__all__ = [
    # Code view
    "StabilizerCode",
    "ROLE_CHECK",
    "ROLE_LOGICAL",
    "ROLE_GAUGE",
    "STRATEGIES",
    # Erasure decoding
    "Candidate",
    "expurgation_candidates",
    "recovery_probability",
    "sample_erasure",
    "uncorrectable_matrix",
    "uncorrectable_rank",
    # Loop driver
    "ExpurgationConfig",
    "ExpurgationResult",
    "expurgate",
    "mean_recovery",
    # Encodings
    "GEOMETRIES",
    "random_encoding",
    # Purification bridge
    "from_purification",
]
