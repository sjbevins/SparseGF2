# Changelog (SparseGF2 v3)

## 3.0.0.dev2: milestone 2, the kernel and `Simulator`

* Column-major, bit-packed tableau kernel (Numba): one 64-bit word holds one
  Pauli bit of 64 generators, so gates are word-parallel and a measurement finds
  all anticommuting generators in one bitset read.
* `Simulator(n, phases=True|False, seed=...)` with Stim-style methods: every
  Clifford gate by name (`sim.h`, `sim.cx`, `sim.iswap`, ...), `do`,
  `clifford_1q/2q` by canonical index, `apply(Clifford)`, `measure`,
  `measure_x/y`, `measure_many`, `measure_pauli` (MPP syntax), `reset`,
  `reset_x/y`, `peek_z/x/y`, `peek_pauli`, `is_deterministic`, `stabilizers`
  (Stim canonical form), `destabilizers`, `stabilizer_arrays`, `entropy`,
  `copy`, `from_tableau`.
* Outcomes are physical in phases mode; phase-free mode returns `None` and
  raises `PhaseFreeError` for sign-dependent queries. `coin=` forces random
  outcomes; every collapse consumes one bit of the xoshiro256** stream.
* Lock-step tests against `stim.TableauSimulator` with forced outcomes: exact
  signed canonical stabilizers after every operation, deterministic outcomes,
  randomness flags, peeks; random initial states; sizes across 64-bit word
  boundaries; plus state-vector checks of the entropy.

## 3.0.0.dev1: milestone 1, foundations

* Audit of v2 (`docs/AUDIT.md`) and the v3 design proposal (`docs/DESIGN.md`).
* `PauliString`: immutable Pauli operators with exact phases, Stim's notation
  and sign conventions; products, commutation, matrices, Stim conversion.
* `Clifford`: k-qubit Clifford unitaries as signed tableaux; conjugation,
  composition, inverse, symplectic matrix, conjugation tables, Stim conversion.
  Canonical, Stim-independent indexing of the 24 one-qubit and 11,520
  two-qubit Cliffords (`Clifford.from_index`, `Clifford.index`,
  `random_clifford_indices`).
* `gate()` / `all_gates()`: every one- and two-qubit Clifford unitary in Stim
  (46 gates plus aliases), identical names and conventions.
* Test suite (156 tests, plus doctests) checking all of the above against Stim, plus
  Stim-independent checks against dense matrices.
* `benchmarks/prototypes/`: the layout prototypes behind the kernel design.
