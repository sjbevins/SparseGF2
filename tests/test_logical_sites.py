"""Weight-one logical operators: the probes ``logical_sites`` (N_log) and ``site_information``
(I(i:R) per qubit), against an exhaustive brute force.

The oracle (``oracle_site_information``) imports nothing from sparsegf2. From the generators of
the pure joint state of the system and the reference R it lists every element of the joint
stabilizer group (all 2^N products), keeps those without support on R (the code's stabilizer
group S, enumerated explicitly), and decides for each system qubit i and P in {X, Y, Z} on i
whether P commutes with every element of S and is not in S: D_i = 1 iff some P does. It also
counts the single-qubit Paulis (identity included) that commute with S, 2^dim N_i, and those
in S, 2^dim S_i, whose log difference is I(i:R) (the cleaning lemma); D_i = 1 iff I(i:R) > 0 is
asserted on every state.

The states come from Stim, never from the engine: trajectories are rebuilt from the documented
draw order (``_protocol_oracle.oracle_draws``) and simulated in ``stim.TableauSimulator``, with
the state compared after every layer, across n <= 8, graphs, schedules, scrambles, both
reference pictures and rates from 0 to 1; hand-built states (k = 0, k = n, single-qubit
stabilizers, Bell pairs inside the system, product states, the repetition code, [[4,2,2]],
[[5,1,3]], [[7,1,3]], [[9,1,3]] and mixed cases) are prepared in Stim by measuring the code's
generators on Bell pairs, and evolved further by gates and measurements in lock step with the
engine's tableau. At larger n (up to 64) the probes are compared on every layer with
contiguous_distance(cap=2) (N_log = 0 iff d >= 2 when k > 0), the full contiguous distance and
the exhaustive one-qubit erasure counts, and runs with the new probes must give bit-identical
trajectories and existing probe values. The summary's conditioning on the codes (mean_with_code,
stderr_with_code, count_with_code, distance_two_fraction) is recomputed from the saved arrays,
with stops, and the codes saved by a snapshot probe give the probes' values through
StabilizerCode.site_information() and logical_sites().
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
import stim

import _protocol_oracle as oracle
from sparsegf2 import Clifford, StabilizerCode, api, graphs, mipt
from sparsegf2._kernel import entropy as ke
from sparsegf2._kernel import tableau as kt
from sparsegf2._kernel.tables import ANF2, IDENTITY_SYM2, LIN2

# ---------------------------------------------------------------------------
# the brute-force oracle (no sparsegf2)
# ---------------------------------------------------------------------------


def _pauli_ints(stabilizers) -> list[int]:
    """Each generator as one integer: X bits of the N qubits in bits 0..N-1, Z bits above."""
    out = []
    for ps in stabilizers:
        xs, zs = ps.to_numpy()
        num = len(xs)
        out.append(
            sum(1 << q for q in np.flatnonzero(xs))
            | sum(1 << (num + q) for q in np.flatnonzero(zs))
        )
    return out


def _group(generators: list[int]) -> np.ndarray:
    """Every product of the generators (2^len of them), as uint64 (N <= 32 qubits)."""
    elements = np.zeros(1, dtype=np.uint64)
    for g in generators:
        elements = np.concatenate([elements, elements ^ np.uint64(g)])
    return elements


def oracle_site_information(stabilizers, n: int) -> tuple[np.ndarray, int]:
    """I(i:R) for the system qubits 0..n-1 by enumeration, and k = n - dim S."""
    num = len(stabilizers[0])
    assert len(stabilizers) == num and num <= 32, "a pure joint state of at most 32 qubits"
    elements = _group(_pauli_ints(stabilizers))
    assert np.unique(elements).size == elements.size, "independent generators"
    reference = sum(1 << q for q in range(n, num)) | sum(1 << (num + q) for q in range(n, num))
    s = elements[(elements & np.uint64(reference)) == 0]  # S: every element without R support
    dim_s = int(round(np.log2(s.size)))
    members = set(s.tolist())
    info = np.zeros(n, dtype=np.int64)
    for i in range(n):
        x_i = (s >> np.uint64(i)) & np.uint64(1)
        z_i = (s >> np.uint64(num + i)) & np.uint64(1)
        commuting = inside = 0
        defect = False
        for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)):  # I, X, Z, Y on qubit i
            # the symplectic product of P = X^a Z^b on i with each element of S
            anti = (np.uint64(a) & z_i) ^ (np.uint64(b) & x_i)
            commutes = not anti.any()
            mask = (a << i) | (b << (num + i))
            in_s = mask in members
            commuting += commutes
            inside += in_s
            if (a, b) != (0, 0) and commutes and not in_s:
                defect = True  # a weight-one logical operator on i
        info[i] = int(round(np.log2(commuting))) - int(round(np.log2(inside)))
        assert defect == (info[i] > 0)
    return info, n - dim_s


# ---------------------------------------------------------------------------
# Stim states, and the same states as the engine's tableau
# ---------------------------------------------------------------------------


@functools.cache
def _stim_clifford(index: int) -> stim.Tableau:
    return Clifford.from_index(2, index).to_stim()


def _stim_trajectory(spec: mipt.MonitoredCircuit, t: int, seed):
    """The canonical stabilizers after setup and scramble (layer 0) and after every layer, from
    the draws rebuilt by the protocol oracle and simulated by Stim."""
    draws = oracle.oracle_draws(spec, t, seed)
    n = spec.num_system
    sim = stim.TableauSimulator()
    sim.set_num_qubits(spec.num_qubits)
    if spec.reference == "purification":
        for i in range(n):
            sim.h(i)
            sim.cnot(i, n + i)
    elif spec.reference == "single":
        sim.h(draws.partner)
        sim.cnot(draws.partner, n)
    for block in draws.scramble:
        for a, b, g in block:
            sim.do_tableau(_stim_clifford(g), [a, b])
    states = [sim.canonical_stabilizers()]
    for gates, measured in zip(draws.gates, draws.measured, strict=True):
        for a, b, g in gates:
            sim.do_tableau(_stim_clifford(g), [a, b])
        for q in measured:
            sim.measure(q)
        states.append(sim.canonical_stabilizers())
    return states


def _engine_tableau(stabilizers, n: int):
    """The engine's column tableau of a pure joint state, with ``ref_rows`` marked.

    The generators are first reduced on the reference columns, so the rows with support on R
    have independent R-parts (the invariant of ``measure_1q_phase_free_tracked``).
    """
    num = len(stabilizers[0])
    rows = np.array([np.concatenate(ps.to_numpy()) for ps in stabilizers], dtype=np.uint8)
    cols = [*range(n, num), *range(num + n, 2 * num)]  # X then Z columns of R
    pivot = 0
    for c in cols:
        hits = np.flatnonzero(rows[pivot:, c])
        if not hits.size:
            continue
        p = pivot + int(hits[0])
        rows[[pivot, p]] = rows[[p, pivot]]
        others = np.flatnonzero(rows[:, c])
        rows[others[others != pivot]] ^= rows[pivot]
        pivot += 1
    ws = (num + 63) // 64
    tab = np.zeros((2 * num + 1, ws), dtype=np.uint64)
    for r in range(num):
        bit = np.uint64(1) << np.uint64(r & 63)
        for q in range(num):
            if rows[r, q]:
                tab[2 * q, r >> 6] |= bit
            if rows[r, num + q]:
                tab[2 * q + 1, r >> 6] |= bit
    ref_rows = np.zeros(ws, dtype=np.uint64)
    ke.reference_rows(tab, ws, np.arange(n, num, dtype=np.int64), ref_rows)
    marked = sum(bin(int(w)).count("1") for w in ref_rows)
    assert marked == pivot  # exactly the reduced rows touch R, with independent R-parts
    return tab, ws, ref_rows


def _kernel_values(tab, ws, n, ref_rows) -> tuple[np.ndarray, int]:
    out = np.zeros(n, dtype=np.int64)
    count = ke.site_information_rows(tab, ws, n, ref_rows, out)
    assert count == ke.site_information_rows(tab, ws, n, ref_rows, np.zeros(0, dtype=np.int64))
    return out, int(count)


def _code_state(stabilizers: list[str]) -> list[stim.PauliString]:
    """A purification of the maximally mixed state of the code: Bell pairs (i, n + i), then a
    measurement of every generator (the signs it picks do not matter)."""
    n = len(stabilizers[0]) if stabilizers else 0
    sim = stim.TableauSimulator()
    sim.set_num_qubits(2 * n)
    for i in range(n):
        sim.h(i)
        sim.cnot(i, n + i)
    for text in stabilizers:
        sim.measure_observable(stim.PauliString(text + "_" * n))
    return sim.canonical_stabilizers()


# ---------------------------------------------------------------------------
# trajectories, every layer, against the oracle
# ---------------------------------------------------------------------------

PROBES = [
    mipt.code_dimension(),
    mipt.logical_sites(),
    mipt.site_information(),
    mipt.contiguous_distance(cap=2),
    mipt.uncorrectable_region_counts(max_size=1),
]


def _ws(n, beta, seed):
    return graphs.watts_strogatz(n, 4, beta, seed=seed)


CASES = [  # name, spec keywords
    ("pair_p0", dict(num_system=2, edges=graphs.path(2), measurement_rate=0.0)),
    ("pair_p03", dict(num_system=2, edges=graphs.path(2), measurement_rate=0.3)),
    ("pair_p1", dict(num_system=2, edges=graphs.path(2), measurement_rate=1.0)),
    ("triangle_single", dict(num_system=3, edges=graphs.cycle(3), measurement_rate=0.2,
                             reference="single")),
    ("ws5_b0", dict(num_system=5, edges=_ws(5, 0.0, 1), measurement_rate=0.1)),
    ("ws6_b03", dict(num_system=6, edges=_ws(6, 0.3, 2), measurement_rate=0.15)),
    ("ws7_b1", dict(num_system=7, edges=_ws(7, 1.0, 3), measurement_rate=0.05)),
    ("ws8_b0", dict(num_system=8, edges=_ws(8, 0.0, 4), measurement_rate=0.2)),
    ("ws8_b1_p0", dict(num_system=8, edges=_ws(8, 1.0, 5), measurement_rate=0.0)),
    ("ws8_b03_p1", dict(num_system=8, edges=_ws(8, 0.3, 6), measurement_rate=1.0)),
    ("ws8_b1_single", dict(num_system=8, edges=_ws(8, 1.0, 7), measurement_rate=0.1,
                           reference="single", scramble_gates=16)),
    ("ring8_matching", dict(num_system=8, edges=graphs.cycle(8), measurement_rate=0.1,
                            schedule="matching")),
    ("path6_all_edges", dict(num_system=6, edges=graphs.path(6), measurement_rate=0.25,
                             schedule="all_edges")),
    ("ws8_scrambled", dict(num_system=8, edges=_ws(8, 0.3, 8), measurement_rate=0.3,
                           scramble_gates=3, scramble_matchings=1, scramble_layers=1)),
    ("ws7_one_gate", dict(num_system=7, edges=_ws(7, 0.3, 9), measurement_rate=0.1,
                          gates_per_layer=1)),
]  # fmt: skip


def _spec(kw, layers=12):
    kw = {"schedule": "random_edges", "reference": "purification", **kw}
    if kw["reference"] == "single":
        kw.setdefault("reference_partner", "random")
    return mipt.MonitoredCircuit(layers=layers, checkpoints="every_layer", probes=PROBES, **kw)


@pytest.mark.parametrize("name, kw", CASES, ids=[c[0] for c in CASES])
def test_every_layer_matches_the_exhaustive_oracle(name, kw):
    spec = _spec(kw)
    seed = [5, len(name)]
    res = spec.run(6, seed=seed)
    k = res.values["code_dimension"]
    n_log = res.values["logical_sites"]
    sites = res.values["site_information"]
    seen = set()  # (k > 0, N_log == 0) over every state compared
    for t in range(res.num_trajectories):
        for layer, state in enumerate(_stim_trajectory(spec, t, seed)):
            info, k_oracle = oracle_site_information(state, spec.num_system)
            assert k[t, layer] == k_oracle, (name, t, layer)
            np.testing.assert_array_equal(sites[t, layer], info, err_msg=f"{name} t={t} L={layer}")
            assert n_log[t, layer] == np.count_nonzero(info)
            seen.add((k_oracle > 0, int(np.count_nonzero(info)) == 0))
    # the one-qubit erasure counts and the cap-2 distance say the same, by other kernels
    np.testing.assert_array_equal(n_log, res.values["uncorrectable_region_counts[1]"][:, :, 0])
    d2 = res.values["contiguous_distance[cap=2]"]
    np.testing.assert_array_equal(np.isnan(d2), k == 0)
    np.testing.assert_array_equal(d2[k > 0] == 2, n_log[k > 0] == 0)
    assert np.all(n_log[k == 0] == 0) and np.all(sites[k == 0] == 0)
    assert (True, False) in seen or name in ("pair_p1", "ws8_b03_p1")  # defects with a code


def test_the_cases_reach_every_kind_of_state():
    """Between them the trajectory cases reach I(i:R) = 0, 1 and 2, k = 0, codes with defects
    and codes with contiguous distance at least 2 (k > 0, N_log = 0)."""
    values: set[float] = set()
    purified = defects_with_code = protected = False
    for name, kw in CASES:
        res = _spec(kw).run(6, seed=[5, len(name)])
        k = res.values["code_dimension"]
        values |= set(np.unique(res.values["site_information"]).tolist())
        purified |= bool((k == 0).any())
        defects_with_code |= bool(((k > 0) & (res.values["logical_sites"] > 0)).any())
        protected |= bool(((k > 0) & (res.values["logical_sites"] == 0)).any())  # d >= 2
    assert values == {0.0, 1.0, 2.0} and purified and defects_with_code and protected


# ---------------------------------------------------------------------------
# hand-built states
# ---------------------------------------------------------------------------

STEANE = ["XXXX___", "XX__XX_", "X_X_X_X", "ZZZZ___", "ZZ__ZZ_", "Z_Z_Z_Z"]
SHOR = [
    "ZZ_______", "_ZZ______", "___ZZ____", "____ZZ___", "______ZZ_", "_______ZZ",
    "XXXXXX___", "___XXXXXX",
]  # fmt: skip
CODES = [  # name, generators, I(i:R) per qubit
    ("k_equals_n", [], None, [2, 2, 2, 2]),
    ("product_k0", ["Z___", "_Z__", "__Z_", "___Z"], None, [0, 0, 0, 0]),
    ("product_x_k0", ["X___", "_Y__", "__Z_", "___X"], None, [0, 0, 0, 0]),
    ("entangled_k0", ["XXX", "ZZ_", "_ZZ"], None, [0, 0, 0]),
    ("single_site_stabilizer", ["Z___"], None, [0, 2, 2, 2]),
    ("bell_pair_in_system", ["XX__", "ZZ__"], None, [0, 0, 2, 2]),
    ("product_and_free", ["X___", "_ZZ_"], None, [0, 1, 1, 2]),
    ("zz_pair", ["ZZ"], None, [1, 1]),
    ("repetition_3", ["ZZ_", "_ZZ"], None, [1, 1, 1]),
    ("parity_3", ["ZZZ"], None, [1, 1, 1]),
    ("code_422", ["XXXX", "ZZZZ"], 2, [0, 0, 0, 0]),
    ("code_422_plus_free", ["XXXX_", "ZZZZ_"], None, [0, 0, 0, 0, 2]),
    ("code_513", ["XZZX_", "_XZZX", "X_XZZ", "ZX_XZ"], 3, [0, 0, 0, 0, 0]),
    ("steane_713", STEANE, 3, [0] * 7),
    ("shor_913", SHOR, 3, [0] * 9),
]


@pytest.mark.parametrize("name, generators, distance, expected", CODES, ids=[c[0] for c in CODES])
def test_hand_built_codes(name, generators, distance, expected):
    n = len(expected)
    state = _code_state(generators) if generators else _code_state(["_" * n])
    info, k = oracle_site_information(state, n)
    np.testing.assert_array_equal(info, expected)
    tab, ws, ref_rows = _engine_tableau(state, n)
    got, count = _kernel_values(tab, ws, n, ref_rows)
    np.testing.assert_array_equal(got, expected)
    assert count == np.count_nonzero(expected)
    assert ke.contiguous_distance_rows_capped(
        tab, ws, np.arange(n, dtype=np.int64), ref_rows, n, True, 2
    ) == (0 if k == 0 else (2 if count == 0 else 1))
    code = StabilizerCode(generators, num_qubits=n)
    assert code.num_logical == k
    np.testing.assert_array_equal(code.site_information(), expected)
    assert code.logical_sites() == count
    if distance is not None:  # known distance >= 2: no single qubit fails, some pair or triple does
        assert all(code.lost_information([i]) == 0 for i in range(n))
        assert code.uncorrectable_counts(distance)[-1] > 0


def test_a_measured_qubit_has_a_local_stabilizer_and_no_logical():
    """Z_i in S after a measurement: local rank 1 at i, but no weight-one logical operator there
    (the case that a local rank test alone gets wrong)."""
    n = 4
    state = _code_state(["Z___"])
    info, _ = oracle_site_information(state, n)
    assert info[0] == 0
    tab, ws, ref_rows = _engine_tableau(state, n)
    # local rank of qubit 0 over the code's stabilizers is 1 (Z_0), over all rows 1 as well
    assert ke.single_qubit_lost(tab, ws, 0, ref_rows) == 0
    assert _kernel_values(tab, ws, n, ref_rows)[0].tolist() == [0, 2, 2, 2]


@pytest.mark.parametrize("start", ["bell", "code_513", "steane_713", "bell_pair_in_system"])
def test_gates_and_measurements_in_lock_step_with_stim(start, rng):
    """Random two-qubit Cliffords (often the same pair again) and single-qubit measurements in
    X, Y and Z, applied to Stim and to the engine's tableau (with its reference-row pivot rule);
    after every operation the kernel equals the oracle."""
    generators = {name: g for name, g, _, _ in CODES}
    if start == "bell":
        n, gens = 6, ["_" * 6]
    else:
        gens = generators[start]
        n = len(gens[0])
    sim = stim.TableauSimulator()
    sim.set_num_qubits(2 * n)
    for i in range(n):
        sim.h(i)
        sim.cnot(i, n + i)
    for text in gens:
        sim.measure_observable(stim.PauliString(text + "_" * n))
    tab, ws, ref_rows = _engine_tableau(sim.canonical_stabilizers(), n)
    num = 2 * n
    anti = np.zeros(ws, dtype=np.uint64)
    supp = np.zeros(num, dtype=np.int64)
    basis = {1: "X", 2: "Z", 3: "Y"}  # the kernel's measurement codes
    pair = (0, 1)
    values = set()
    for step in range(120):
        if rng.random() < 0.55:
            if rng.random() < 0.5:  # a repeated gate selection: the same pair again
                pair = tuple(int(q) for q in rng.choice(n, 2, replace=False))
            g = int(rng.integers(11520))
            sim.do_tableau(_stim_clifford(g), list(pair))
            kt.apply_2q(tab, pair[0], pair[1], g, LIN2, ANF2, IDENTITY_SYM2, ws, False)
        else:
            q = int(rng.integers(n))
            code = int(rng.integers(1, 4))
            sim.measure_observable(stim.PauliString("_" * q + basis[code] + "_" * (num - q - 1)))
            kt.measure_1q_phase_free_tracked(tab, num, ws, q, code, anti, supp, ref_rows)
        info, k = oracle_site_information(sim.canonical_stabilizers(), n)
        got, count = _kernel_values(tab, ws, n, ref_rows)
        np.testing.assert_array_equal(got, info, err_msg=f"{start} step {step}")
        assert sum(bin(int(w)).count("1") for w in ref_rows) - n == k
        values.update(info.tolist())
        if k == 0:
            break
    assert len(values) >= 2


# ---------------------------------------------------------------------------
# larger n: against the cap-2 and full contiguous distances and the exhaustive counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [16, 32, 64])
@pytest.mark.parametrize("beta", [0.0, 0.3, 1.0])
def test_no_logical_site_iff_the_distance_reaches_two(n, beta):
    """On every layer of many trajectories: N_log = 0 iff min(d_cont, 2) = 2 where k > 0, N_log
    is the number of uncorrectable one-qubit erasures, sum(I > 0) = N_log, and (n <= 32) the full
    contiguous distance is 1 exactly where N_log > 0."""
    probes = [
        mipt.code_dimension(),
        mipt.logical_sites(),
        mipt.site_information(),
        mipt.contiguous_distance(cap=2),
        mipt.uncorrectable_region_counts(max_size=1, name="one"),
    ]
    if n <= 32:
        probes.append(mipt.contiguous_distance())
    seen_zero = seen_defects = 0
    for p in (0.02, 0.05, 0.1, 0.2):
        spec = mipt.MonitoredCircuit(
            num_system=n,
            edges=[graphs.watts_strogatz(n, 4, beta, seed=(31, g)) for g in range(3)],
            circuits_per_graph=3,
            layers=2 * n,
            measurement_rate=p,
            schedule="random_edges",
            reference="purification",
            checkpoints="every_layer",
            probes=probes,
        )
        res = spec.run(seed=[n, int(10 * beta), int(100 * p)])
        k = res.values["code_dimension"]
        n_log = res.values["logical_sites"]
        d2 = res.values["contiguous_distance[cap=2]"]
        code = k > 0
        np.testing.assert_array_equal(d2[code] == 2, n_log[code] == 0)
        np.testing.assert_array_equal(n_log, res.values["one"][:, :, 0])
        np.testing.assert_array_equal(n_log, (res.values["site_information"] > 0).sum(axis=2))
        assert np.all(n_log[~code] == 0)
        if n <= 32:
            d = res.values["contiguous_distance"]
            np.testing.assert_array_equal(d[code] == 1, n_log[code] > 0)
        seen_zero += int((code & (n_log == 0)).sum())
        seen_defects += int((code & (n_log > 0)).sum())
    assert seen_zero > 0 and seen_defects > 0


def test_the_single_reference_picture():
    """With one reference qubit N_log counts the qubits carrying a weight-one logical of the
    one logical qubit; I(i:R) is 0, 1 or 2 there too."""
    n = 32
    spec = mipt.MonitoredCircuit(
        num_system=n, edges=graphs.watts_strogatz(n, 4, 1.0, seed=2), layers=64,
        measurement_rate=0.1, schedule="random_edges", reference="single",
        reference_partner="random", checkpoints="every_layer",
        probes=[mipt.code_dimension(), mipt.logical_sites(), mipt.contiguous_distance(cap=2),
                mipt.uncorrectable_region_counts(max_size=1, name="one")],
    )  # fmt: skip
    res = spec.run(24, seed=3)
    k, n_log = res.values["code_dimension"], res.values["logical_sites"]
    np.testing.assert_array_equal(n_log, res.values["one"][:, :, 0])
    d2 = res.values["contiguous_distance[cap=2]"]
    np.testing.assert_array_equal(d2[k > 0] == 2, n_log[k > 0] == 0)
    assert np.all(n_log[:, 0] == 1)  # the partner, before any gate
    assert np.all(n_log[k == 0] == 0)


# ---------------------------------------------------------------------------
# the new probes change nothing else
# ---------------------------------------------------------------------------


def _base_probes():
    return [
        mipt.code_dimension(),
        mipt.half_cut_entropy(checkpoints=[0, 5, 40]),
        mipt.contiguous_distance(cap=2),
        mipt.erasure_first_failure(samples=7, checkpoints=[3, 30]),
        mipt.tableau_snapshot(checkpoints=[0, 20, 40]),
    ]


@pytest.mark.parametrize("stop", [None, "purified", "distance"])
@pytest.mark.parametrize("picture", ["purification", "single"])
def test_runs_with_the_new_probes_are_bit_identical(stop, picture):
    n = 12
    kw = dict(
        num_system=n,
        edges=[graphs.watts_strogatz(n, 4, 0.3, seed=(5, g)) for g in range(2)],
        circuits_per_graph=6,
        layers=40,
        measurement_rate=0.15,
        schedule="random_edges",
        reference=picture,
        checkpoints="every_layer",
        probes=_base_probes(),
        stop_when_purified=stop == "purified",
    )
    if picture == "single":
        kw["reference_partner"] = "random"
    if stop == "distance":
        kw.update(stop_when_distance=2, stop_not_before=5)
    base = mipt.MonitoredCircuit(**kw)
    more = mipt.MonitoredCircuit(
        **{
            **kw,
            "probes": [
                mipt.site_information(checkpoints=[0, 7, 40]),
                *_base_probes(),
                mipt.logical_sites(),
            ],
            "record_first_touch": True,
        }
    )
    a, b = base.run(seed=11), more.run(seed=11)
    for name in a.values:
        np.testing.assert_array_equal(a.values[name], b.values[name])
    for field in ("purification_layer", "final_layer", "num_measurements", "first_passage"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    for name, (words, counts) in a.snapshots.items():
        np.testing.assert_array_equal(words, b.snapshots[name][0])
        np.testing.assert_array_equal(counts, b.snapshots[name][1])
    # the same circuits: the replayed circuit text does not depend on the probes
    assert str(base.circuit(3, seed=11)) == str(more.circuit(3, seed=11))
    # after a stop the new probes take the fills of the other code probes
    final = b.final_layer
    for t in range(b.num_trajectories):
        after = np.arange(41) > final[t]
        values = b.values["logical_sites"][t, after]
        if stop == "distance" and b.stopped_on_distance[t]:
            assert np.isnan(values).all()
        elif stop == "purified" and b.purification_layer[t] >= 0:
            assert np.all(values == 0)


def test_after_a_purification_stop_the_new_probes_record_zero():
    spec = mipt.MonitoredCircuit(
        num_system=8, edges=graphs.cycle(8), layers=60, measurement_rate=0.4,
        schedule="random_edges", reference="purification", checkpoints="every_layer",
        stop_when_purified=True,
        probes=[mipt.logical_sites(), mipt.site_information(checkpoints=[0, 30, 60])],
    )  # fmt: skip
    res = spec.run(10, seed=2)
    stopped = res.purification_layer >= 0
    assert stopped.any()
    for t in np.flatnonzero(stopped):
        tau = res.purification_layer[t]
        assert np.all(res.values["logical_sites"][t, tau:] == 0)
        later = res.layers("site_information") >= tau
        assert np.all(res.values["site_information"][t, later] == 0)


def test_probe_constructors_and_validation():
    assert mipt.logical_sites().name == "logical_sites"
    assert mipt.logical_sites("nlog").name == "nlog"
    p = mipt.site_information(checkpoints=[0, 4])
    assert p.name == "site_information" and p.is_vector and p.checkpoints == (0, 4)
    assert not mipt.logical_sites().is_vector
    for probe in (mipt.logical_sites(), mipt.site_information()):
        with pytest.raises(ValueError, match="needs a reference"):
            mipt.MonitoredCircuit(num_system=4, edges=graphs.cycle(4), layers=2,
                                  measurement_rate=0.1, probes=[probe])  # fmt: skip
    spec = mipt.MonitoredCircuit(num_system=5, edges=graphs.cycle(5), layers=3,
                                 measurement_rate=0.1, reference="purification",
                                 probes=[mipt.site_information()])  # fmt: skip
    assert spec.run(2, seed=1).values["site_information"].shape == (2, 1, 5)


def test_saved_loaded_and_the_json_api(tmp_path):
    spec = {
        "num_system": 16,
        "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.3, "seed": 1},
        "layers": 24,
        "measurement_rate": 0.1,
        "schedule": "random_edges",
        "reference": "purification",
        "checkpoints": 4,
        "probes": [
            {"kind": "code_dimension"},
            {"kind": "logical_sites"},
            {"kind": "site_information", "checkpoints": [0, 12, 24]},
        ],
    }
    out = api.run_ensemble(spec, trajectories=30, seed=2, output_dir=str(tmp_path / "run"))
    res = mipt.EnsembleResult.load(tmp_path / "run" / "result.npz")
    n_log = res.values["logical_sites"]
    k = res.values["code_dimension"]
    entry = out["probes"]["logical_sites"]
    np.testing.assert_allclose(entry["mean"], n_log.mean(axis=0), rtol=1e-5)
    assert entry["count_with_code"] == (k > 0).sum(axis=0).tolist()
    np.testing.assert_allclose(
        entry["distance_two_fraction"], ((k > 0) & (n_log == 0)).mean(axis=0), rtol=1e-5
    )
    assert any(note.startswith("logical_sites: N_log") for note in out["notes"])
    sites = out["probes"]["site_information"]
    assert sites["vector"] and len(sites["mean"]) == 16 and "I(i:R)" in sites["what"]
    np.testing.assert_array_equal(
        (res.values["site_information"] > 0).sum(axis=2), n_log[:, [0, 3, 6]]
    )
    # without a code_dimension probe at its layers the summary says how to get the conditioning
    alone = api.run_ensemble(
        {**spec, "probes": [{"kind": "logical_sites"}]}, trajectories=5, seed=2
    )
    assert "mean_with_code" not in alone["probes"]["logical_sites"]
    assert any("add a code_dimension probe" in note for note in alone["notes"])
    caps = api.capabilities("probes")["probes"]
    assert {"logical_sites", "site_information"} <= set(caps)
    estimate = api.estimate_ensemble(spec, trajectories=10)
    names = [row["name"] for row in estimate["layout"]["arrays"]]
    assert "value::site_information" in names
    row = next(r for r in estimate["layout"]["arrays"] if r["name"] == "value::site_information")
    assert row["shape"] == [10, 3, 16]
    # priced like the cap-2 distance on a code without defects, far below the full distance
    mc = api._build_spec(spec)
    cheap = api._probe_us(mipt.logical_sites(), mc)
    assert cheap == api._probe_us(mipt.contiguous_distance(cap=2), mc)
    assert cheap < api._probe_us(mipt.contiguous_distance(), mc)


# ---------------------------------------------------------------------------
# the summary's conditioning on the codes, from the saved arrays
# ---------------------------------------------------------------------------

CONDITIONED_SPEC = {
    "num_system": 12,
    "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.3, "seed": 2, "realizations": 4},
    "circuits_per_graph": 6,
    "layers": 30,
    "measurement_rate": 0.2,
    "schedule": "random_edges",
    "reference": "purification",
    "checkpoints": "every_layer",
    "probes": [
        {"kind": "code_dimension"},
        {"kind": "logical_sites"},
        {"kind": "contiguous_distance", "cap": 2},
    ],
}
CONDITIONED_CASES = [  # name, spec changes
    ("no_stop", {}),
    ("purification_stop", {"stop_when_purified": True}),
    ("distance_stop", {"measurement_rate": 0.1, "stop_when_distance": 2, "stop_not_before": 4}),
    # logical_sites on layers of its own, covered by a code_dimension probe at every layer
    ("own_layers", {"probes": [{"kind": "code_dimension"},
                               {"kind": "logical_sites", "checkpoints": [0, 2, 5, 9, 17, 30]}]}),
    # no code_dimension probe: the codes are read off erasure_first_failure (D <= n iff k > 0)
    ("first_failure", {"probes": [{"kind": "erasure_first_failure", "samples": 20},
                                  {"kind": "logical_sites"}]}),
    # both stops with one reference qubit: columns with one code (no stderr) and with none
    ("single_both_stops", {"reference": "single", "reference_partner": "random",
                           "measurement_rate": 0.1, "stop_when_purified": True,
                           "stop_when_distance": 2}),
]  # fmt: skip


def _shown(value, expected: float) -> bool:
    """A summary number (6 significant digits, NaN shown as None) against its exact value."""
    if np.isnan(expected):
        return value is None
    return value == pytest.approx(expected, rel=1e-5, abs=1e-12)


def _holds_a_code(res: mipt.EnsembleResult, layers: list[int]) -> np.ndarray:
    """k > 0 per row at ``layers``, from the code_dimension probe (NaN after a distance stop is no
    code), or without one from erasure_first_failure: some first failure at most n."""
    n = res.spec.num_system
    if "code_dimension" in res.values:
        own = res.layers("code_dimension").tolist()
        k = res.values["code_dimension"][:, [own.index(L) for L in layers]]
        return np.nan_to_num(k, nan=0.0) > 0
    own = res.layers("erasure_first_failure").tolist()
    first = res.values["erasure_first_failure"][:, [own.index(L) for L in layers]]
    return (np.nan_to_num(first, nan=np.inf) <= n).any(axis=2)


@pytest.mark.parametrize("name, changes", CONDITIONED_CASES, ids=[c[0] for c in CONDITIONED_CASES])
def test_the_summary_conditions_on_the_codes(tmp_path, name, changes):
    """mean_with_code and stderr_with_code (the sample standard deviation over sqrt(count), from
    analysis.mean_and_error) over the rows with k > 0 whose N_log is known (not NaN after a
    distance stop), their number count_with_code, and distance_two_fraction, the rows with k > 0
    and N_log = 0 over the rows whose N_log is known, at every shown layer."""
    spec = {**CONDITIONED_SPEC, **changes}
    out = api.run_ensemble(spec, seed=4, output_dir=str(tmp_path / "run"))
    res = mipt.EnsembleResult.load(tmp_path / "run" / "result.npz")
    entry = out["probes"]["logical_sites"]
    layers = entry.get("layers", out["checkpoints"]["layers"])
    own = res.layers("logical_sites").tolist()
    assert layers == own  # every evaluation is shown here
    n_log = res.values["logical_sites"]
    holds = _holds_a_code(res, layers)
    for j in range(len(layers)):
        known = ~np.isnan(n_log[:, j])
        codes = n_log[holds[:, j] & known, j]
        assert entry["count_with_code"][j] == codes.size
        assert _shown(entry["mean_with_code"][j], codes.mean() if codes.size else np.nan)
        error = np.std(codes, ddof=1) / np.sqrt(codes.size) if codes.size > 1 else np.nan
        assert _shown(entry["stderr_with_code"][j], error)
        protected = np.sum(holds[:, j] & known & (n_log[:, j] == 0))
        fraction = protected / known.sum() if known.any() else np.nan
        assert _shown(entry["distance_two_fraction"][j], fraction)
        # the unconditioned mean counts k = 0 as 0 and leaves out the unknown rows
        assert _shown(entry["mean"][j], n_log[known, j].mean() if known.any() else np.nan)
    assert min(entry["count_with_code"]) < res.num_trajectories  # the conditioning matters
    assert any("mean_with_code averages" in note for note in out["notes"])
    if name == "distance_stop":
        assert np.isnan(n_log).any()
    if name == "single_both_stops":
        assert 0 in entry["count_with_code"] and 1 in entry["count_with_code"]


# ---------------------------------------------------------------------------
# the saved codes give the probes' values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("snapshot", ["code_snapshot", "tableau_snapshot"])
@pytest.mark.parametrize("stop", [None, "purified", "distance"])
@pytest.mark.parametrize("picture", ["purification", "single"])
def test_saved_codes_give_the_probe_values(picture, stop, snapshot):
    """EnsembleResult.code(t, c).site_information() and .logical_sites() equal the probes at the
    same evaluation wherever a code was saved (all 0 where k = 0); where the row had stopped
    nothing was saved, code() raises, and the probes hold 0 (pure reference) or NaN (distance)."""
    n = 10
    at = [0, 3, 8, 16, 24]
    kw = dict(
        num_system=n,
        edges=[_ws(n, 0.3, (17, g)) for g in range(3)],
        circuits_per_graph=6,
        layers=24,
        measurement_rate=0.2 if picture == "purification" else 0.08,
        schedule="random_edges",
        reference=picture,
        checkpoints=at,
        probes=[
            mipt.code_dimension(),
            mipt.logical_sites(),
            mipt.site_information(),
            getattr(mipt, snapshot)(),
            mipt.contiguous_distance(cap=2, checkpoints="every_layer"),
        ],
        stop_when_purified=stop == "purified",
    )
    if picture == "single":
        kw["reference_partner"] = "random"
    if stop == "distance":
        kw.update(stop_when_distance=2, stop_not_before=3)
    res = mipt.MonitoredCircuit(**kw).run(seed=[6, len(picture)])
    k = res.values["code_dimension"]
    n_log = res.values["logical_sites"]
    sites = res.values["site_information"]
    saved = res.snapshots[snapshot][1]
    seen = set()
    for t in range(res.num_trajectories):
        for c, layer in enumerate(at):
            if saved[t, c] < 0:  # the row had stopped: no state, no code
                assert layer > res.final_layer[t]
                with pytest.raises(ValueError, match="had stopped"):
                    res.code(t, c)
                fill = np.nan if res.stopped_on_distance[t] else 0.0
                np.testing.assert_array_equal(n_log[t, c], fill)
                np.testing.assert_array_equal(sites[t, c], np.full(n, fill))
                seen.add("stopped")
                continue
            code = res.code(t, c)
            assert code.num_qubits == n and code.num_logical == k[t, c], (t, c)
            np.testing.assert_array_equal(code.site_information(), sites[t, c], err_msg=f"{t} {c}")
            assert code.logical_sites() == n_log[t, c], (t, c)
            if k[t, c] == 0:  # no logical qubit: no information to lose at any site
                assert n_log[t, c] == 0 and not sites[t, c].any()
                seen.add("no code")
            else:
                seen.add("weight-one logicals" if n_log[t, c] > 0 else "d >= 2")
    assert {"weight-one logicals", "d >= 2"} <= seen
    assert ("stopped" in seen) == (stop is not None)
    if stop != "distance":
        assert "no code" in seen
