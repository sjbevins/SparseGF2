"""Independent references for phase-free row spaces (test-only; imports nothing from sparsegf2).

A row is a Python int whose bit ``j`` is column ``j`` in the canonical order
``x_0 .. x_{N-1}, z_0 .. z_{N-1}`` (``x_q`` is bit ``q``, ``z_q`` is bit ``N + q``). Random
stabilizer states come from Stim, driven by gates chosen with numpy, so a fixed numpy seed
fixes them.
"""

from __future__ import annotations

import numpy as np
import stim


def int_rows(dense) -> list[int]:
    """The rows of a 0/1 matrix as ints (bit j = column j)."""
    return [sum(1 << int(j) for j in np.flatnonzero(row)) for row in np.asarray(dense)]


def dense_rows(rows: list[int], width: int) -> np.ndarray:
    out = np.zeros((len(rows), width), dtype=np.uint8)
    for i, r in enumerate(rows):
        for j in range(width):
            out[i, j] = (r >> j) & 1
    return out


def rref(rows: list[int], order) -> list[int]:
    """Reduced row echelon form for the column order ``order`` (Gauss-Jordan on ints).

    A row's pivot is its first column in ``order`` holding a 1. Each new pivot row is cleared
    from every other row; the nonzero rows are returned in pivot order.
    """
    remaining = list(rows)
    basis: list[int] = []
    for col in order:
        bit = 1 << int(col)
        pick = next((i for i, r in enumerate(remaining) if r & bit), None)
        if pick is None:
            continue
        p = remaining.pop(pick)
        remaining = [r ^ p if r & bit else r for r in remaining]
        basis = [b ^ p if b & bit else b for b in basis]
        basis.append(p)
    return basis


def canonical(dense) -> np.ndarray:
    """The canonical form of the rows of a ``(s, 2N)`` 0/1 matrix, as a 0/1 matrix."""
    width = np.asarray(dense).shape[1]
    return dense_rows(rref(int_rows(dense), range(width)), width)


def rank(rows: list[int]) -> int:
    """GF(2) rank of int rows: each row is reduced by the kept rows keyed by their highest
    bit, and kept if something is left (a different elimination order from ``rref``)."""
    kept: dict[int, int] = {}
    for v in rows:
        while v:
            top = v.bit_length() - 1
            if top not in kept:
                kept[top] = v
                break
            v ^= kept[top]
    return len(kept)


def restrict(g: int, qubits, n: int) -> int:
    """The Pauli g on n qubits (x_q bit q, z_q bit n + q) restricted to ``qubits`` (in the
    given order) as a Pauli on len(qubits) qubits."""
    qubits = list(qubits)
    m = len(qubits)
    out = 0
    for i, q in enumerate(qubits):
        out |= ((g >> q) & 1) << i
        out |= ((g >> (n + q)) & 1) << (m + i)
    return out


def physical_rows(rows: list[int], n: int, num_qubits: int) -> list[int]:
    """The canonical rows of S_phys of a joint state (system 0..n-1, reference n..N-1).

    Gauss-Jordan with the reference columns first (``rref`` with that order): the rows whose
    lead is a system column are zero on the reference (they are the elements of the space with
    no reference part); restricted to the system and brought to the canonical order.
    """
    big = num_qubits
    reference = [q for q in range(n, big)] + [big + q for q in range(n, big)]
    order = reference + list(range(n)) + [big + q for q in range(n)]
    ref_mask = sum(1 << c for c in reference)
    system_part = [r for r in rref(rows, order) if r & ref_mask == 0]
    return rref([restrict(r, range(n), big) for r in system_part], range(2 * n))


def span(rows: list[int]) -> set[int]:
    """Every element of the row space (2**rank of them)."""
    out = {0}
    for r in rows:
        out |= {s ^ r for s in out}
    return out


def rref_by_characterization(rows: list[int]) -> list[int]:
    """The canonical basis from its defining property alone, by enumerating the space.

    The leading columns are the lowest set bits of the nonzero elements; for each leading
    column p the canonical row is the one element whose lowest set bit is p and which is zero
    at every other leading column (it exists and is unique: the proof is in
    docs/study/03-phase-free-tableau.md). Exponential: small spaces only.
    """
    elements = span(rows)
    leads = sorted({(v & -v).bit_length() - 1 for v in elements if v})
    mask = sum(1 << p for p in leads)
    out = []
    for p in leads:
        found = [v for v in elements if v & mask == 1 << p and v & ((1 << p) - 1) == 0]
        assert len(found) == 1
        out.append(found[0])
    return out


def support(g: int, n: int) -> int:
    """The qubits on which the Pauli g (x bits 0..n-1, z bits n..2n-1) acts, as a bit mask."""
    return (g | (g >> n)) & ((1 << n) - 1)


def log2_exact(count: int) -> int:
    assert count > 0 and count & (count - 1) == 0, count
    return count.bit_length() - 1


def entropy_by_enumeration(elements: set[int], n: int, region) -> int:
    """S(A) = |A| - log2 #{g : supp g within A}, over the elements of the stabilizer group."""
    mask = sum(1 << int(q) for q in region)
    inside = sum(1 for g in elements if support(g, n) & ~mask == 0)
    return len(list(region)) - log2_exact(inside)


def symplectic(a: int, b: int, n: int) -> int:
    """The symplectic product x.z' + z.x' (mod 2) of two Paulis on n qubits."""
    low = (1 << n) - 1
    return (bin(a & low & (b >> n)).count("1") + bin((a >> n) & b & low).count("1")) & 1


def pack(dense: np.ndarray, n: int) -> np.ndarray:
    """The engine packing, bit by bit: x_q at bit q % 64 of word q // 64, z_q in word wq + q // 64."""
    wq = (n + 63) // 64
    out = np.zeros((dense.shape[0], 2 * wq), dtype=np.uint64)
    for i, row in enumerate(dense):
        for q in range(n):
            for half in range(2):
                if row[half * n + q]:
                    out[i, half * wq + q // 64] |= np.uint64(1) << np.uint64(q % 64)
    return out


def unpack(words: np.ndarray, n: int) -> np.ndarray:
    wq = (n + 63) // 64
    out = np.zeros((words.shape[0], 2 * n), dtype=np.uint8)
    for i in range(words.shape[0]):
        for q in range(n):
            for half in range(2):
                out[i, half * n + q] = (int(words[i, half * wq + q // 64]) >> (q % 64)) & 1
    return out


def packed_position(column: int, n: int) -> int:
    """Where canonical column ``column`` (x_q = q, z_q = n + q) sits in the engine packing."""
    wq = (n + 63) // 64
    return column if column < n else 64 * wq + (column - n)


def scramble(sim: stim.TableauSimulator, qubits, rng, depth: int) -> None:
    """``depth`` rounds of a CX on a random pair and an H, S or nothing on every qubit."""
    qubits = list(qubits)
    for _ in range(depth):
        if len(qubits) > 1:
            a, b = (int(q) for q in rng.choice(qubits, size=2, replace=False))
            sim.cx(a, b)
        for q in qubits:
            gate = rng.integers(4)
            if gate == 0:
                sim.h(q)
            elif gate == 1:
                sim.s(q)


def bits(p: stim.PauliString) -> np.ndarray:
    """A Stim Pauli string as its (x | z) row, sign dropped."""
    xs, zs = p.to_numpy()
    return np.concatenate([xs, zs]).astype(np.uint8)


def stim_state(num_qubits: int, rng) -> np.ndarray:
    """The unsigned stabilizers of a random pure stabilizer state, from Stim: (N, 2N)."""
    sim = stim.TableauSimulator(seed=int(rng.integers(2**31)))
    sim.set_num_qubits(num_qubits)
    scramble(sim, range(num_qubits), rng, 4 * num_qubits + 4)
    return np.array([bits(p) for p in sim.canonical_stabilizers()])


def stim_joint_state(n: int, rng, measured, pairs=None) -> np.ndarray:
    """Bell pairs of the system qubits ``pairs`` (default all n; the i-th with reference
    n + i), a random circuit on the system with Z measurements of ``measured`` in the middle:
    the unsigned stabilizers over n + len(pairs) qubits."""
    pairs = list(range(n)) if pairs is None else list(pairs)
    sim = stim.TableauSimulator(seed=int(rng.integers(2**31)))
    sim.set_num_qubits(n + len(pairs))
    for i, q in enumerate(pairs):
        sim.h(q)
        sim.cx(q, n + i)
    scramble(sim, range(n), rng, 3 * n + 2)
    for q in measured:
        sim.measure(int(q))
    scramble(sim, range(n), rng, 3 * n + 2)
    return np.array([bits(p) for p in sim.canonical_stabilizers()])


def stim_tick_states(circuit: stim.Circuit, num_qubits: int, seed: int) -> list[np.ndarray]:
    """The unsigned stabilizers after every TICK of ``circuit`` in Stim's TableauSimulator."""
    sim = stim.TableauSimulator(seed=seed)
    sim.set_num_qubits(num_qubits)
    states = []
    for ins in circuit.flattened():
        if ins.name == "TICK":
            states.append(np.array([bits(p) for p in sim.canonical_stabilizers()]))
        else:
            sim.do(ins)
    return states
