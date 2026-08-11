"""Role bookkeeping: the (S, L, G) code view over a SparseGF2 tableau.

An encoded stabilizer tableau *is* a code: after any Clifford
construction from :math:`|0^N\\rangle`, stabilizer row ``N + i`` holds
:math:`U Z_i U^\\dagger` and destabilizer row ``i`` holds
:math:`U X_i U^\\dagger`, and the canonical destabilizer/stabilizer
pairing is exactly the code algebra (checks commute with everything,
each logical pair anticommutes within the pair and commutes across
pairs). The only extra data a code needs is *which role each pair
plays*:

* ``ROLE_CHECK`` (S): the pair's stabilizer row is a check.
* ``ROLE_LOGICAL`` (L): the pair carries a logical qubit
  (:math:`\\bar{Z}` on the stabilizer row, :math:`\\bar{X}` on the
  destabilizer row).
* ``ROLE_GAUGE`` (G): the pair is gauge, i.e. an expurgated logical
  whose rows are kept but protect nothing (Poulin's stabilizer
  subsystem codes, `quant-ph/0508131 <https://arxiv.org/abs/0508131>`_).

:class:`StabilizerCode` bundles a simulator with that role array and
implements the elementary expurgation move: projectively measure a
zero-syndrome Pauli with nontrivial logical action, then flip the role
of the pair that absorbed it. Expurgation never relabels tableau rows;
measurement updates the rows and the role array is the only code-level
state (Gullans et al., PRX 11, 031066 (2021), Sec. VI).

The class is agnostic to how the tableau was generated: encoding
circuits built from core gates, :meth:`SparseGF2.from_symplectic` on an
externally produced tableau, or circuits-layer output all work, because
the only coupling is the simulator's measurement and commutation API
plus the pair-index convention shared by every ``SparseGF2``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self

import numpy as np
from numpy.typing import NDArray

from sparsegf2.core.sparse_tableau import SparseGF2
from sparsegf2.errors import InvalidArgumentError

# Role codes. Stored in a uint8 array of length ``sim.n`` indexed by
# pair (pair ``i`` = destabilizer row ``i`` + stabilizer row ``n + i``).
ROLE_CHECK = np.uint8(0)
ROLE_LOGICAL = np.uint8(1)
ROLE_GAUGE = np.uint8(2)

#: Accepted ``strategy`` values for :meth:`StabilizerCode.measure`.
STRATEGIES = ("stabilizer", "gauge")


def _exact_integer_array(values: Iterable[int], *, name: str) -> NDArray[np.int64]:
    """Return a 1-D int64 array without accepting lossy numeric coercions."""
    raw = np.asarray(list(values), dtype=object)
    if raw.ndim != 1:
        raise InvalidArgumentError(f"{name} must be one-dimensional; got shape {raw.shape}")
    no_bad_value = object()
    bad = next(
        (
            value
            for value in raw
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        ),
        no_bad_value,
    )
    if bad is not no_bad_value:
        raise InvalidArgumentError(
            f"{name} must contain exact integers (not Booleans or lossy numeric values); "
            f"got {bad!r}"
        )
    return np.asarray([int(value) for value in raw], dtype=np.int64)


class StabilizerCode:
    """A stabilizer (subsystem) code presented as a tableau plus roles.

    Parameters
    ----------
    sim
        A :class:`sparsegf2.SparseGF2` holding the encoded tableau. The
        instance is used in place (not copied); measurements through
        :meth:`measure` mutate it.
    roles
        Length-``sim.n`` array over pair indices with values
        :data:`ROLE_CHECK`, :data:`ROLE_LOGICAL`, :data:`ROLE_GAUGE`.
        Copied on construction.

    Notes
    -----
    The constructor validates shape and values only. It cannot validate
    that the labeling is *physically* consistent (that every check-row
    really commutes with the rest of the state's structure is a property
    of how the tableau was built); :meth:`measure` re-checks the one
    consequence expurgation relies on and fails loudly if the labels lie.
    """

    def __init__(self, sim: SparseGF2, roles: Iterable[int]):
        role_values = _exact_integer_array(roles, name="roles")
        role_arr = role_values.astype(np.uint8, copy=True)
        if role_arr.shape != (sim.n,):
            raise InvalidArgumentError(
                f"roles must have shape ({sim.n},) = (sim.n,), got {role_arr.shape}"
            )
        if role_values.size and ((role_values < 0).any() or (role_values > 2).any()):
            raise InvalidArgumentError(
                "roles entries must be ROLE_CHECK (0), ROLE_LOGICAL (1), or ROLE_GAUGE (2); "
                f"got values {sorted(set(role_values.tolist()))}"
            )
        self.sim = sim
        self.roles: NDArray[np.uint8] = role_arr

    @classmethod
    def from_encoding(cls, sim: SparseGF2, data_qubits: Iterable[int]) -> Self:
        """Build the code view for the standard encoding convention.

        The Gullans et al. codes place the :math:`k` logical inputs on a
        set ``K`` of qubits, freeze the rest to :math:`|0\\rangle`, and
        apply the encoding circuit. Run that circuit on a
        ``SparseGF2(n)`` (which starts in :math:`|0^n\\rangle`) and the
        tableau is the code: pairs in ``K`` are logical, the rest are
        checks.

        Parameters
        ----------
        sim
            The simulator after the encoding circuit.
        data_qubits
            The set ``K`` of logical input positions, distinct indices
            in ``[0, sim.n)``.
        """
        data = _exact_integer_array(data_qubits, name="data_qubits")
        if data.size:
            if data.min() < 0 or data.max() >= sim.n:
                raise InvalidArgumentError(
                    f"data_qubits must be in [0, n={sim.n}); got min={data.min()}, max={data.max()}"
                )
            if np.unique(data).shape[0] != data.shape[0]:
                raise InvalidArgumentError(f"data_qubits must not repeat: got {data.tolist()}")
        role_arr = np.full(sim.n, ROLE_CHECK, dtype=np.uint8)
        role_arr[data] = ROLE_LOGICAL
        return cls(sim, role_arr)

    # ------------------------------------------------------------------
    # Structure queries
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of physical qubits."""
        return self.sim.n

    @property
    def k(self) -> int:
        """Number of logical pairs still protected."""
        return int(np.count_nonzero(self.roles == ROLE_LOGICAL))

    def check_pairs(self) -> NDArray[np.int64]:
        """Sorted pair indices with role S (checks)."""
        return np.flatnonzero(self.roles == ROLE_CHECK)

    def logical_pairs(self) -> NDArray[np.int64]:
        """Sorted pair indices with role L (protected logical qubits)."""
        return np.flatnonzero(self.roles == ROLE_LOGICAL)

    def gauge_pairs(self) -> NDArray[np.int64]:
        """Sorted pair indices with role G (expurgated / gauge pairs)."""
        return np.flatnonzero(self.roles == ROLE_GAUGE)

    def copy(self) -> Self:
        """Deep copy: snapshots the simulator and the role array."""
        return type(self)(self.sim.copy(), self.roles)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n={self.n}, k={self.k}, "
            f"checks={len(self.check_pairs())}, gauge={len(self.gauge_pairs())})"
        )

    # ------------------------------------------------------------------
    # Commutation bits
    # ------------------------------------------------------------------

    def commutation_bits(
        self, qubits: Iterable[int], letters: Iterable[int]
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Syndrome and logical-action bits of a Pauli against this code.

        For a Pauli ``E`` given sparsely (support ``qubits``, 2-bit
        ``letters``), returns

        * ``syndrome``: commutation bits with the check generators, one
          per pair in :meth:`check_pairs` order. ``E`` is undetectable
          exactly when this is all zero.
        * ``logical``: commutation bits with the logical generators,
          two per pair in :meth:`logical_pairs` order: first against
          the pair's :math:`\\bar{Z}` (stabilizer row), then against its
          :math:`\\bar{X}` (destabilizer row). ``E`` acts nontrivially
          on the protected qubits exactly when this is nonzero.

        Gauge pairs contribute no bits, which is the subsystem-code
        convention (their columns are removed from the uncorrectable
        error matrix).

        Both arrays are computed from one commutation query on the
        simulator, so this works in sparse and hybrid dense mode alike.
        """
        n = self.n
        anti = self.sim.pauli_anticommuting_rows(qubits, letters)
        mask = np.zeros(2 * n, dtype=bool)
        mask[anti] = True
        checks = self.check_pairs()
        logicals = self.logical_pairs()
        syndrome = mask[n + checks].astype(np.uint8)
        logical = np.empty(2 * logicals.shape[0], dtype=np.uint8)
        logical[0::2] = mask[n + logicals]
        logical[1::2] = mask[logicals]
        return syndrome, logical

    # ------------------------------------------------------------------
    # The elementary expurgation move
    # ------------------------------------------------------------------

    def measure(
        self, qubits: Iterable[int], letters: Iterable[int], *, strategy: str = "gauge"
    ) -> int | None:
        """Measure a zero-syndrome Pauli and spend the pair that absorbs it.

        The elementary move of the expurgation algorithm: project the
        code space onto an eigenspace of ``g`` by consuming a logical
        pair on which ``g`` acts nontrivially. Under the ``"stabilizer"``
        strategy the pair becomes
        a new check (the code shrinks to a plain stabilizer code with
        one more check); under ``"gauge"`` it becomes a gauge pair (the
        original checks are untouched, the paper's preferred variant).

        Parameters
        ----------
        qubits, letters
            The Pauli ``g`` in the sparse convention of
            :meth:`SparseGF2.measure_pauli`. Must have zero syndrome
            against the current checks; measuring a detectable operator
            is not an expurgation move and raises.
        strategy
            ``"stabilizer"`` or ``"gauge"``.

        Returns
        -------
        int or None
            The logical pair index whose role was flipped, or ``None``
            when ``g`` acts trivially on the current logical algebra. A
            skipped move leaves the tableau and the roles unchanged.
        """
        if strategy not in STRATEGIES:
            raise InvalidArgumentError(f"strategy={strategy!r} not in {STRATEGIES}")
        syndrome, logical = self.commutation_bits(qubits, letters)
        if syndrome.any():
            raise InvalidArgumentError(
                "measure: the operator anticommutes with a check; expurgation only "
                "measures zero-syndrome operators (re-validate candidates first)"
            )
        if not logical.any():
            return None
        logical_pairs = self.logical_pairs()
        logical_slot = int(np.flatnonzero(logical)[0]) // 2
        pair = int(logical_pairs[logical_slot])
        self.sim.project_pauli_into_pair(qubits, letters, pair)
        self.roles[pair] = ROLE_CHECK if strategy == "stabilizer" else ROLE_GAUGE
        return pair
