"""Known-answer tests for the randomness contract (``mipt.RANDOMNESS_CONTRACT``).

The kernel generator is compared with the pure-Python reference in ``_reference_rng``
(an independent transcription of xoshiro256**, splitmix64 and Lemire's method, with the
stream derivation of every contract), and literal values pin the stream keys, the
circuits an ensemble draws, a sampled-probe stream and a small run. The shot streams of
``sample`` and the streams of the sampled observables are checked against the reference.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import _reference_rng as ref
from sparsegf2 import Circuit, Simulator, graphs, mipt, observables, sample
from sparsegf2._kernel import rng as krng


def _state(words) -> np.ndarray:
    return np.array([int(w) for w in words], dtype=np.uint64)


def test_published_xoshiro_vector():
    # xoshiro256** from state [1, 2, 3, 4] (the published values); the first output is
    # rotl(2 * 5, 7) * 9 = 1280 * 9 = 11520.
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
    # next_below. Only a bound that does not divide 2**32 and is large (here 3 * 2**30,
    # threshold 2**30, about 25% rejection per draw) reaches the rejection loop.
    for bound in (1, 2, 3, 11520, 2**31 - 1, 3 * 2**30):
        s, r = _state(words), ref.Xoshiro(words)
        for _ in range(draws):
            got = int(krng.next_below(s, bound))
            assert got == r.next_below(bound)
            assert 0 <= got < bound
        if bound == 3 * 2**30:
            # ~250 expected rejections in 1000 draws; the loop really ran
            assert r.rejections > 100
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
        want = ref.stream_state(key, index, contract=mipt.RANDOMNESS_CONTRACT)
        assert [int(w) for w in out] == want, (key, index)


def test_mix_equals_reference():
    gen = np.random.default_rng(2025)
    top = 2**64 - 1
    triples = [(0, 0, 0), (top, top, top), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2**40, top, 7)]
    while len(triples) < 50:
        triples.append(tuple(int(w) for w in gen.integers(0, 2**64, size=3, dtype=np.uint64)))
    for a, b, c in triples:
        assert int(krng.mix(np.uint64(a), np.uint64(b), np.uint64(c))) == ref.mix(a, b, c)
    # the engine passes the trajectory and the layer as int64
    assert int(krng.mix(np.int64(12), np.uint64(5), np.int64(3))) == ref.mix(12, 5, 3)


def test_stream_state_uses_the_whole_key():
    # Kernel finding S1: contract 1 kept only key0 ^ key1, so these keys shared every
    # stream. Contract 2 separates them, and a different index gives a different state.
    a, b = (0x0123456789ABCDEF, 0xFEDCBA9876543210), (0x0123456789ABCDEE, 0xFEDCBA9876543211)
    assert a[0] ^ a[1] == b[0] ^ b[1]
    assert ref.stream_state(a, 3, contract=1) == ref.stream_state(b, 3, contract=1)
    out_a, out_b = np.zeros(4, dtype=np.uint64), np.zeros(4, dtype=np.uint64)
    krng.stream_state(_state(a), 3, out_a)
    krng.stream_state(_state(b), 3, out_b)
    assert not np.array_equal(out_a, out_b)
    for key in (a, b, (0, 0)):
        states = {tuple(ref.stream_state(key, i, contract=2)) for i in range(64)}
        assert len(states) == 64
    # every word changes when any one key bit or index bit does (a raw key1 added to the
    # increment would leave words 1 and 3 unchanged when its top bit flips)
    base = ref.stream_state(a, 9, contract=2)
    for bit in range(128):
        flipped = (a[0] ^ (1 << bit), a[1]) if bit < 64 else (a[0], a[1] ^ (1 << (bit - 64)))
        other = ref.stream_state(flipped, 9, contract=2)
        assert all(u != v for u, v in zip(base, other, strict=True)), f"key bit {bit}"
    for bit in range(64):
        other = ref.stream_state(a, 9 ^ (1 << bit), contract=2)
        assert all(u != v for u, v in zip(base, other, strict=True)), f"index bit {bit}"


def test_sample_shot_streams_follow_the_reference():
    # Every M of a qubit in |+> is random and its outcome is its coin: the next bit of the
    # shot's stream. Shot s of sample(seed=k) reads stream s of stream_key(k).
    qubits = " ".join(str(q) for q in range(10))
    result = sample(Circuit(f"H {qubits}\nM {qubits}"), 6, seed=2024)
    words = np.random.SeedSequence(2024).generate_state(2, dtype=np.uint64)
    key = (int(words[0]), int(words[1]))
    for shot in range(6):
        r = ref.Xoshiro(ref.stream_state(key, shot, contract=mipt.RANDOMNESS_CONTRACT))
        assert result.measurements[shot].tolist() == [r.next_bit() for _ in range(10)], shot


def test_sampled_observables_follow_the_reference():
    # One reference (qubit 6) Bell-paired with system qubit 2: an erasure loses 2 bits
    # exactly when it holds qubit 2, so the lost information shows which subsets and
    # orderings were drawn from stream 0 of stream_key(seed).
    n, partner, sim = 6, 2, Simulator(7, phases=False)
    sim.run(Circuit(f"H {partner}\nCX {partner} {n}"))
    words = np.random.SeedSequence(31).generate_state(2, dtype=np.uint64)
    key = (int(words[0]), int(words[1]))
    sizes, samples = [1, 3, 5], 7
    stats = observables.random_erasure_statistics(
        sim, range(n), [n], sizes=sizes, samples=samples, seed=31
    )
    r = ref.Xoshiro(ref.stream_state(key, 0, contract=mipt.RANDOMNESS_CONTRACT))
    perm, want = list(range(n)), []
    for m in sizes:  # a partial Fisher-Yates per sample, continuing from the last order
        row = []
        for _ in range(samples):
            for t in range(m):
                j = t + r.next_below(n - t)
                perm[t], perm[j] = perm[j], perm[t]
            row.append(2 if partner in perm[:m] else 0)
        want.append(row)
    assert stats["lost"].tolist() == want
    assert len({v for row in want for v in row}) == 2  # both outcomes occur
    profile = observables.erasure_profile(sim, range(n), [n], samples=samples, seed=31)
    r = ref.Xoshiro(ref.stream_state(key, 0, contract=mipt.RANDOMNESS_CONTRACT))
    perm, want = list(range(n)), []
    for _ in range(samples):  # a full Fisher-Yates per sample, of the previous ordering
        for t in range(n - 1):
            j = t + r.next_below(n - t)
            perm[t], perm[j] = perm[j], perm[t]
        want.append([2 if partner in perm[:m] else 0 for m in range(n + 1)])
    assert profile["lost"].tolist() == want


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
    # Contract 1 (3.0.0.dev11 and before): kept to document it. Its stream_state is still
    # checked against the reference's contract-1 mixing; the package no longer draws these.
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
    # Contract 2: all 128 key bits in every stream; one stream per sampled-probe evaluation.
    2: {
        "stream_state": [
            16914043347802755243,
            6493749712075630968,
            12956986678988623689,
            17403317311179609507,
        ],
        "probe_stream": [
            281794202721889763,
            5210048329218925464,
            3313951943835708634,
            381803185464458647,
        ],
        "circuits": {
            "S1/t0": "aafc9d4809f2a71578933faf76be80e319445f74f290933e281b492206d9695c",
            "S1/t3": "63b0873a876d6af22e9aa344fe741907fc33d8479f903153136721b535d78cd7",
            "S2/t0": "fd17183f51cfbb14a30f5d217a7559cd9217c4e258c5f4e3174ecc45c9f3bfbb",
            "S2/t3": "c1fb5dc6d58eb90e4a2c1ede146935f183860b8f1ca6438ac408bfdf38b238ce",
            "S3/t0": "d9df9cc7ac9def5a723dcf071e7c526f1610531907dc945a2ed65a965c648372",
            "S3/t3": "5d96dcb22250fe3a4072b1a0221db2a1dec0f879da55fc5b7209837e5df36724",
        },
        "run": "6755400726570ee83559961a33b6d56881ee5294b3ea38fa6548fb6e3096da28",
    },
}

# The sampled-probe evaluation pinned by "probe_stream": run(seed=11), trajectory 5,
# probe "erasure_first_failure", layer 4.
PROBE_STREAM = (11, 5, "erasure_first_failure", 4)


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


def _probe_stream_from_package() -> list[int]:
    seed, t, name, layer = PROBE_STREAM
    _, key_probe = mipt._keys(np.random.SeedSequence(seed))
    out = np.zeros(4, dtype=np.uint64)
    index = krng.mix(np.int64(t), np.uint64(mipt._name_hash(name)), np.int64(layer))
    krng.stream_state(key_probe, np.uint64(index), out)
    return [int(w) for w in out]


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
        "probe_stream": _probe_stream_from_package(),
        "circuits": circuits,
        "run": h.hexdigest(),
    }


def test_reference_stream_state_is_pinned_value():
    key = (0x0123456789ABCDEF, 0xFEDCBA9876543210)
    assert ref.RANDOMNESS_CONTRACT_MODEL == mipt.RANDOMNESS_CONTRACT
    assert set(EXPECTED) == set(ref.CONTRACTS)
    for contract, values in EXPECTED.items():
        assert ref.stream_state(key, 7, contract=contract) == values["stream_state"], contract


def test_reference_probe_stream_is_pinned_value():
    # The engine's stream for one sampled-probe evaluation (randomness contract 2),
    # transcribed: words 2 and 3 of the run's SeedSequence key it, and its index is
    # mix(trajectory, blake2b-8 of the name read little-endian, layer).
    seed, t, name, layer = PROBE_STREAM
    words = np.random.SeedSequence(seed).generate_state(4, dtype=np.uint64)
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    index = ref.mix(t, int.from_bytes(digest, "little"), layer)
    state = ref.stream_state((int(words[2]), int(words[3])), index, contract=2)
    assert state == EXPECTED[2]["probe_stream"]


def test_pinned_values_for_the_current_contract():
    assert mipt.RANDOMNESS_CONTRACT in EXPECTED, (
        "RANDOMNESS_CONTRACT changed: add an entry to EXPECTED (never edit an old one)"
    )
    assert _current_values() == EXPECTED[mipt.RANDOMNESS_CONTRACT]
