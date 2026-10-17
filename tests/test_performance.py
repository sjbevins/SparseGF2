"""The 720 symplectic classes of the phase-free gates, and CPU-specific settings that change no
result: threads, placements, presets, the tableau padding and the sweep's chunk rounds.

Statistics: one chi-square test at ``ALPHA = 1e-6``; a correct sampler at a fresh seed fails
it with probability 1e-6 (at the fixed seed here the outcome is deterministic).
"""

from __future__ import annotations

import dataclasses
import json
import sys

import numba
import numpy as np
import pytest

import _ws_reference as wsref
from sparsegf2 import cli, mipt, performance
from sparsegf2._kernel import ensemble as _ens
from sparsegf2._kernel.tableau import apply_2q
from sparsegf2._kernel.tables import ANF2, IDENTITY_SYM2, LIN2
from sparsegf2.clifford import symplectic_group
from sparsegf2.performance import CacheDomain, CpuTopology, PerformanceSettings
from sparsegf2.study import identity, sweep
from sparsegf2.study.sweep import SweepSpec

ALPHA = 1e-6
TOPOLOGY = performance.cpu_topology()
PLACEABLE = TOPOLOGY.known and sys.platform in ("win32", "linux")


def condition(n=12, p="0.2", graphs=3, **changes):
    """The sweep's engine for one condition (purification picture, snapshots) and its seed."""
    spec = SweepSpec.from_json(
        {
            "sizes": [n],
            "betas": ["0.1"],
            "rates": [p],
            "graphs": graphs,
            "circuits_per_graph": 4,
            "stride": 4,
            "layers": {"per_n": 3},
            "checkpoints": {"count": 5},
            "master_graph_seed": 17,
            "master_circuit_seed": 29,
            **changes,
        }
    )
    collection = [identity.build_graph(n, "0.1", g, 17).edges for g in range(graphs)]
    return spec.monitored_circuit(n, "0.1", p, collection), list(spec.run_seed(n, "0.1"))


def assert_same(a: mipt.EnsembleResult, b: mipt.EnsembleResult) -> None:
    """Every value and snapshot word equal, and every snapshot's canonical digest."""
    assert np.array_equal(a.trajectories, b.trajectories)
    for k in a.values:
        assert np.array_equal(a.values[k], b.values[k], equal_nan=True), k
    for k in a.snapshots:
        assert np.array_equal(a.snapshots[k][0], b.snapshots[k][0]), k
        assert np.array_equal(a.snapshots[k][1], b.snapshots[k][1]), k
    for name in ("purification_layer", "final_layer", "num_measurements"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    for e in range(len(a.layers("tableau_snapshot"))):
        da = [t.digest() for t in a.tableaux(e)]
        assert da == [t.digest() for t in b.tableaux(e)]


# ---------------------------------------------------------------------------
# the 720 classes
# ---------------------------------------------------------------------------


def test_the_kernel_applies_every_index_as_the_symplectic_matrix_of_its_class():
    group = symplectic_group(2).astype(np.uint64)
    assert np.array_equal(LIN2 != 0, group != 0)  # LIN2[s] holds M_s as all-ones masks
    rng = np.random.default_rng(5)
    base = rng.integers(0, 2**63, size=(2 * 5 + 1, 1), dtype=np.int64).astype(np.uint64)
    a, b = 1, 3
    bits = np.unpackbits(base[[2 * a, 2 * b, 2 * a + 1, 2 * b + 1], 0].view(np.uint8))
    v = bits.reshape(4, 64)  # rows x_a, x_b, z_a, z_b; one column per generator (bit order)
    for j in range(11520):
        tab, rep = base.copy(), base.copy()
        apply_2q(tab, a, b, j, LIN2, ANF2, IDENTITY_SYM2, 1, False)
        apply_2q(rep, a, b, 16 * (j >> 4), LIN2, ANF2, IDENTITY_SYM2, 1, False)
        assert np.array_equal(tab, rep)  # the sign bits j & 15 act on the sign row only
        if j % 16 == 0:  # (x_a, x_b, z_a, z_b) <- v M_s (mod 2) for all 64 generators
            m = group[j >> 4].astype(np.int64)
            image = (v.T.astype(np.int64) @ m % 2).T.astype(np.uint8)
            words = np.packbits(image.reshape(-1)).view(np.uint64)
            assert np.array_equal(tab[[2 * a, 2 * b, 2 * a + 1, 2 * b + 1], 0], words)
            untouched = [r for r in range(11) if r not in (2 * a, 2 * b, 2 * a + 1, 2 * b + 1)]
            assert np.array_equal(tab[untouched], base[untouched])


def test_a_run_with_every_index_replaced_by_its_class_representative_is_identical():
    mc, seed = condition()
    rep = dataclasses.replace(mc, gate_set=16 * (np.arange(11520, dtype=np.int64) >> 4))
    assert rep.gate_set.size == 11520  # the same pool size: the same draws, index for index
    assert_same(mc.run(12, seed=seed), rep.run(12, seed=seed))


def test_the_engine_draws_the_720_classes_uniformly():
    n, layers = 16, 512
    mc, seed = condition(n=n, layers={str(n): layers})
    a = mc._arrays()
    key_struct, _ = mipt._keys(mipt._seed_sequence(seed))
    classes = []
    for u in range(25):  # the draws run_ensemble makes: n // 2 gates per layer
        ga = np.zeros(layers * (n // 2), dtype=np.int64)
        gb, gg = np.zeros_like(ga), np.zeros_like(ga)
        _ens.generate_ops(
            n, 1, a["partner"], a["edges"], a["graph_ptr"], mc.circuits_per_graph,
            a["edge_cum"], a["color_ptr"], a["color_ptr_ptr"], a["color_idx"], 1, a["gpl"],
            a["pool"], float(mc.measurement_rate), layers, 0, 0, 0, False, key_struct, u, ga,
            gb, gg, np.zeros(layers + 2, dtype=np.int64), np.zeros(layers * n, dtype=np.int64),
            np.zeros(layers + 1, dtype=np.int64),
        )  # fmt: skip
        classes.append(gg >> 4)
    counts = np.bincount(np.concatenate(classes), minlength=720)
    assert counts.sum() == 25 * layers * (n // 2) == 102_400 and counts.size == 720
    assert wsref.uniform_p(counts)[2] > ALPHA  # 142 expected per class, 719 degrees of freedom


# ---------------------------------------------------------------------------
# settings change no result
# ---------------------------------------------------------------------------


def _available(setting):
    if isinstance(setting, PerformanceSettings) and setting.placement != "all":
        if not PLACEABLE:
            pytest.skip("thread placement needs a known topology on Windows or Linux")
        if setting.placement in ("vcache", "other_ccd") and not TOPOLOGY.x3d:
            pytest.skip("not a 3D V-Cache part: no L3 domains of different sizes")
    if setting == "9950x3d" and not performance._is_9950x3d(TOPOLOGY):
        pytest.skip(f"not a Ryzen 9 9950X3D: {performance._describe(TOPOLOGY)}")
    return setting


SETTINGS = [
    PerformanceSettings(threads=1),
    PerformanceSettings(threads=2),
    PerformanceSettings(),
    PerformanceSettings(placement="physical"),
    PerformanceSettings(placement="physical", threads=3),
    PerformanceSettings(placement="vcache"),
    PerformanceSettings(placement="other_ccd", threads=5),
    PerformanceSettings(placement=TOPOLOGY.processors[:2] or (0,)),
    "auto",
    "9950x3d",
]


@pytest.fixture(scope="module")
def default_run():
    mc, seed = condition()
    return mc, seed, mc.run(12, seed=seed)


@pytest.mark.parametrize("setting", SETTINGS, ids=str)
def test_every_setting_gives_identical_results(default_run, setting):
    mc, seed, expected = default_run
    assert_same(mc.run(12, seed=seed, performance=_available(setting)), expected)


@pytest.mark.parametrize("pad", [0, 1, 3])
def test_the_tableau_padding_changes_no_result(default_run, monkeypatch, pad):
    mc, seed, expected = default_run
    monkeypatch.setattr(mipt, "_row_padding", lambda words: pad)
    assert_same(mc.run(12, seed=seed), expected)


def test_the_padding_used_from_16_words_changes_no_result(monkeypatch):
    # N = 1024 qubits (n = 512 and its references): 16 words per column, padded to 17
    assert [mipt._row_padding(w) for w in (1, 2, 8, 15, 16, 24, 32, 48, 64)] == [
        0, 0, 0, 0, 1, 0, 1, 1, 1,
    ]  # fmt: skip
    mc, seed = condition(n=512, graphs=1, layers={"512": 6})
    padded = mc.run(4, seed=seed)
    monkeypatch.setattr(mipt, "_row_padding", lambda words: 0)
    assert_same(mc.run(4, seed=seed), padded)


def test_a_sweep_with_a_preset_and_whole_rounds_equals_the_default(tmp_path):
    spec = {
        "sizes": [6, 8],
        "betas": ["0.1"],
        "rates": ["0.2"],
        "graphs": 3,
        "circuits_per_graph": 3,
        "stride": 4,
        "layers": {"per_n": 2},
        "checkpoints": {"count": 4},
        "master_graph_seed": 17,
        "master_circuit_seed": 29,
    }
    sweep.run(spec, tmp_path / "default", chunk_size=4)
    sweep.run(spec, tmp_path / "auto", performance="auto")
    settings = PerformanceSettings(threads=2, writers=1)
    sweep.run(spec, tmp_path / "two", performance=settings)
    for n in (6, 8):
        expected = sweep.load(tmp_path / "default", n, "0.1", "0.2")
        for name in ("auto", "two"):
            assert_same(sweep.load(tmp_path / name, n, "0.1", "0.2"), expected)
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    argv = ["sweep", "run", str(spec_file), str(tmp_path / "cli"), "--performance", "auto"]
    assert cli.main([*argv, "--json"]) == 0
    assert_same(sweep.load(tmp_path / "cli", 8, "0.1", "0.2"), expected)


def test_chunks_are_whole_rounds_of_the_running_threads():
    assert sweep._rounds(100, 1000, 32) == 96  # rounded down to whole rounds
    assert sweep._rounds(10, 1000, 32) == 32  # at least one round
    assert sweep._rounds(5000, 40, 32) == 32  # within the cap
    assert sweep._rounds(64, 64, 32) == 64 and sweep._rounds(7, 7, 1) == 7


# ---------------------------------------------------------------------------
# presets, topology and affinity
# ---------------------------------------------------------------------------


def x3d_like() -> CpuTopology:
    cores = tuple((2 * i, 2 * i + 1) for i in range(16))
    l3 = (CacheDomain(96 * 2**20, tuple(range(16))), CacheDomain(32 * 2**20, tuple(range(16, 32))))
    return CpuTopology("windows", tuple(range(32)), cores, l3, 2**20, 48 * 2**10)


def test_the_presets_follow_the_cache_rule():
    topo = x3d_like()
    assert performance._is_9950x3d(topo) and topo.x3d and topo.vcache == topo.l3[0]
    # one thread per core once 32 tableaux exceed the largest L3, 96 MiB: from N = 3521 qubits
    largest = 96 * 2**20
    assert 32 * performance.tableau_bytes(3520) <= largest < 32 * performance.tableau_bytes(3521)
    for name in ("9950x3d", "auto"):
        for qubits, placement in ((128, "all"), (3072, "all"), (3520, "all"), (3521, "physical")):
            assert performance.preset(name, qubits, topo).placement == placement
    assert topo.placement("physical") == tuple(range(0, 32, 2))
    assert topo.placement("other_ccd") == tuple(range(16, 32))
    single = CpuTopology("linux", tuple(range(8)), tuple((i,) for i in range(8)),
                         (CacheDomain(32 * 2**20, tuple(range(8))),))  # fmt: skip
    with pytest.raises(ValueError, match="use 'auto'"):
        performance.preset("9950x3d", 128, single)
    assert performance.preset("auto", 64000, single).placement == "physical"
    with pytest.raises(ValueError, match="an X3D part"):
        single.placement("vcache")
    unknown = CpuTopology("unknown")
    assert performance.preset("auto", 10**6, unknown) == PerformanceSettings()
    with pytest.raises(ValueError, match="preset must be one of"):
        performance.preset("fastest", 128, topo)
    with pytest.raises(ValueError, match="threads must be a positive integer"):
        PerformanceSettings(threads=0)


def test_the_topology_of_this_machine():
    assert performance.cpu_topology() == TOPOLOGY  # never raises; the same answer again
    if not TOPOLOGY.x3d:
        pytest.skip(f"not a 3D V-Cache part (two L3 sizes): {performance._describe(TOPOLOGY)}")
    big, small = sorted(TOPOLOGY.l3, key=lambda d: -d.size)
    assert len(TOPOLOGY.l3) == 2 and big.size > small.size  # the V-Cache CCD and the other
    assert set(big.processors).isdisjoint(small.processors)
    assert set(big.processors) | set(small.processors) == set(TOPOLOGY.processors)
    assert TOPOLOGY.l2_bytes and TOPOLOGY.l1d_bytes
    assert len(TOPOLOGY.placement("physical")) == len(TOPOLOGY.cores)


@pytest.mark.skipif(
    not PLACEABLE, reason="thread placement needs a known topology on Windows or Linux"
)
def test_the_affinity_and_thread_count_are_restored_also_on_an_exception():
    before, threads = performance._get_affinity(), numba.get_num_threads()
    physical = TOPOLOGY.placement("physical")
    with performance.applied(PerformanceSettings(placement="physical")):
        assert performance._get_affinity() == physical
        assert numba.get_num_threads() == min(len(physical), numba.config.NUMBA_NUM_THREADS)
    assert performance._get_affinity() == before and numba.get_num_threads() == threads
    one = PerformanceSettings(placement=physical[:1], threads=1)
    with pytest.raises(RuntimeError, match="inside"), performance.applied(one):
        assert performance._get_affinity() == physical[:1]
        raise RuntimeError("inside")
    assert performance._get_affinity() == before and numba.get_num_threads() == threads
