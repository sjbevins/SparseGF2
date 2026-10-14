"""Known-answer tests for the randomness contract (``mipt.RANDOMNESS_CONTRACT``).

The kernel generator is compared with the pure-Python reference in ``_reference_rng``
(an independent transcription of xoshiro256**, splitmix64 and Lemire's method), and
literal values pin the stream keys, the circuits an ensemble draws and a small run.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import _reference_rng as ref
from sparsegf2 import graphs, mipt
from sparsegf2._kernel import rng as krng


def _state(words) -> np.ndarray:
    return np.array([int(w) for w in words], dtype=np.uint64)


def test_published_xoshiro_vector():
    # xoshiro256** from state [1, 2, 3, 4]; the first three values follow by hand from
    # rotl(s1 * 5, 7) * 9.
    expected = [11520, 0, 1509978240, 1215971899390074240]
    r = ref.Xoshiro([1, 2, 3, 4])
    assert [r.next_u64() for _ in expected] == expected
    s = _state([1, 2, 3, 4])
    assert [int(krng.next_u64(s)) for _ in expected] == expected


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kernel_equals_reference(seed):
    words = [int(w) for w in krng.seed_state(seed)]
    draws = 1000
    # next_u64
    s, r = _state(words), ref.Xoshiro(words)
    for _ in range(draws):
        assert int(krng.next_u64(s)) == r.next_u64()
    # next_double (bit-identical: both are an integer below 2**53 times 2**-53)
    s, r = _state(words), ref.Xoshiro(words)
    for _ in range(draws):
        assert float(krng.next_double(s)) == r.next_double()
    # next_bit
    s, r = _state(words), ref.Xoshiro(words)
    for _ in range(draws):
        assert int(krng.next_bit(s)) == r.next_bit()
    # next_below, including bounds that force rejections' threshold branch
    for bound in (1, 2, 3, 11520, 2**31 - 1):
        s, r = _state(words), ref.Xoshiro(words)
        for _ in range(draws):
            got = int(krng.next_below(s, bound))
            assert got == r.next_below(bound)
            assert 0 <= got < bound
        # the two generators consumed the same number of draws
        assert int(krng.next_u64(s)) == r.next_u64()


def test_stream_state_equals_reference():
    gen = np.random.default_rng(2024)
    pairs = [((0, 0), 0), ((2**64 - 1, 2**64 - 1), 2**40), ((5, 9), 0), ((5, 9), 2**40)]
    while len(pairs) < 50:
        key = tuple(int(w) for w in gen.integers(0, 2**63, size=2, dtype=np.uint64) * 2 + 1)
        index = int(gen.integers(0, 10**6)) if len(pairs) % 2 else 2**40 + len(pairs)
        pairs.append((key, index))
    out = np.zeros(4, dtype=np.uint64)
    for key, index in pairs:
        krng.stream_state(_state(key), index, out)
        assert [int(w) for w in out] == ref.stream_state(key, index), (key, index)


def test_stream_key_matches_seed_sequence():
    key = krng.stream_key(5)
    assert key.dtype == np.uint64 and key.shape == (2,)
    assert [int(w) for w in key] == [
        int(w) for w in np.random.SeedSequence(5).generate_state(2, dtype=np.uint64)
    ]


# -- pinned values -----------------------------------------------------------------------
#
# A changed value means the random draws changed. Bump `RANDOMNESS_CONTRACT` and add a new
# entry. Never edit an existing one.
EXPECTED: dict[int, dict] = {
    1: {
        "stream_state": [
            11467144058010140927,
            14501507772988823413,
            7777348718182911552,
            11091782217626042368,
        ],
        "circuits": {
            "S1/t0": "bed0820928164d091f8fb1a6a75069be6eb4b770da19e6574932488b2b5e5176",
            "S1/t3": "111f3e27a7d6f4604cd1a38813d4321f76115c70798cf7f6f2e012ad47b4c8b3",
            "S2/t0": "3dd007ec773f0f6e1f085f6c38f99276fd89416322a5452104b8b8a3a5756c06",
            "S2/t3": "f113aadb26f1c2952b19ac573c7788f59024c95391b9d4f680fe5b39cb5b4fa5",
            "S3/t0": "33236f7ce2397df157956377820b6ba1a216d66a40b8881042a93e201ccc9a4f",
            "S3/t3": "7fc0cc75187d9913c4bb9966d692b382678eaac07750ac63ce574c99c6dc16e9",
        },
        "run": "bd92422ad271ef8b783026e83f9e9f001b5cbcb4ac42f051fb3a75dbce1ef4ef",
    },
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _specs() -> dict[str, mipt.MonitoredCircuit]:
    s1 = mipt.MonitoredCircuit(
        num_system=8,
        edges=graphs.cycle(8),
        layers=4,
        measurement_rate=0.2,
        schedule="matching",
        reference="purification",
        scramble_gates=3,
    )
    collection = [graphs.watts_strogatz(10, 4, 0.3, seed=(1, i)) for i in range(2)]
    s2 = mipt.MonitoredCircuit(
        num_system=10,
        edges=collection,
        circuits_per_graph=2,
        layers=3,
        measurement_rate=0.15,
        schedule="random_edges",
        edge_weights=[np.linspace(1.0, 2.0, len(e)) for e in collection],
        random_orientation=True,
        reference="single",
        reference_partner="random",
        scramble_matchings=1,
        scramble_layers=1,
    )
    s3 = mipt.MonitoredCircuit(
        num_system=6,
        edges=graphs.path(6),
        layers=2,
        measurement_rate=0.3,
        schedule="all_edges",
        gate_set=np.array([0, 17, 400, 5000, 11519]),
        reference="none",
    )
    return {"S1": s1, "S2": s2, "S3": s3}


def _current_values() -> dict:
    key = (0x0123456789ABCDEF, 0xFEDCBA9876543210)
    out = np.zeros(4, dtype=np.uint64)
    krng.stream_state(_state(key), 7, out)
    circuits = {}
    for name, spec in _specs().items():
        for t in (0, 3):
            if spec.num_trajectories is not None and t >= spec.num_trajectories:
                continue
            circuits[f"{name}/t{t}"] = _sha(str(spec.circuit(t, seed=2024)))
    s1 = _specs()["S1"]
    spec = mipt.MonitoredCircuit(
        num_system=s1.num_system,
        edges=s1.edges,
        layers=s1.layers,
        measurement_rate=s1.measurement_rate,
        schedule=s1.schedule,
        reference=s1.reference,
        scramble_gates=s1.scramble_gates,
        checkpoints=[0, 2, 4],
        probes=[
            mipt.code_dimension(),
            mipt.erasure_first_failure(samples=8),
            mipt.uncorrectable_region_counts(max_size=2),
        ],
    )
    res = spec.run(6, seed=11)
    h = hashlib.sha256()
    for name in sorted(res.values):
        h.update(name.encode("utf-8"))
        h.update(np.ascontiguousarray(res.values[name]).astype("<f8").tobytes())
    h.update(np.ascontiguousarray(res.purification_layer).astype("<i8").tobytes())
    h.update(np.ascontiguousarray(res.num_measurements).astype("<i8").tobytes())
    return {
        "stream_state": [int(w) for w in out],
        "circuits": circuits,
        "run": h.hexdigest(),
    }


def test_reference_stream_state_is_pinned_value():
    key = (0x0123456789ABCDEF, 0xFEDCBA9876543210)
    assert ref.RANDOMNESS_CONTRACT_MODEL in EXPECTED
    assert ref.stream_state(key, 7) == EXPECTED[ref.RANDOMNESS_CONTRACT_MODEL]["stream_state"]


def test_pinned_values_for_the_current_contract():
    assert mipt.RANDOMNESS_CONTRACT in EXPECTED, (
        "RANDOMNESS_CONTRACT changed: add an entry to EXPECTED (never edit an old one)"
    )
    assert _current_values() == EXPECTED[mipt.RANDOMNESS_CONTRACT]
