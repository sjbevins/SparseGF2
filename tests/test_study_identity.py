"""Identities, seeds, graph records and canonical bytes of the Watts–Strogatz study.

Covers ``sparsegf2.study.identity`` (design §3, v1.4 §2, §7.2, §8): exact rate tokens, the
trajectory key, the seed hierarchy, the stream index, graph records and their digests, and
canonical JSON. Everything here is deterministic; the statistical conformance of the graph
and circuit sampling is in ``test_study_sampling.py``. Expected values are computed in the
tests from the definitions (``fractions``, ``decimal``, ``hashlib``, ``struct``), not by the
code under test, and golden vectors were computed from the independent reference in
``_ws_reference.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import _ws_reference as wsref
from sparsegf2 import api, graphs, mipt
from sparsegf2.study import identity as ident
from sparsegf2.study.identity import (
    GraphRecord,
    Token,
    TrajectoryKey,
    build_graph,
    canonical_json,
    circulant_edges,
    edge_digest,
    graph_seed,
    json_digest,
    original_edge_list,
    rate_hash,
    run_seed,
    sha256_hex,
    split_stream_index,
    stream_index,
    validate_graph,
)

PROPERTY = settings(max_examples=300, deadline=None, database=None, derandomize=True)

# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling",
    ["0.1", 0.1, "1/10", ".1", "1e-1", "0.10", "00.1000", "+0.1", " 0.1 ", Fraction(1, 10), "2/20"],
)
def test_equivalent_spellings_give_one_token(spelling):
    token = Token.parse(spelling)
    assert token == Token.parse("0.1")
    assert token.token == "0.1" and str(token) == "0.1"
    assert token.value == Fraction(1, 10)
    assert hash(token) == hash(Token.parse("1/10"))
    assert Token.parse(token) is token


@pytest.mark.parametrize(
    "spelling, canonical",
    [
        ("0", "0"),
        (0, "0"),
        ("0.0", "0"),
        ("-0", "0"),
        ("1", "1"),
        (1, "1"),
        ("1.", "1"),
        ("1.000", "1"),
        ("5/5", "1"),
        ("0.5", "0.5"),
        ("1/2", "0.5"),
        ("0.25", "0.25"),
        ("1/20", "0.05"),
        ("3/8", "0.375"),
        ("1e-2", "0.01"),
        ("5e-1", "0.5"),
        (1e-05, "0.00001"),
        ("1/1000000", "0.000001"),
        (0.30000000000000004, "0.30000000000000004"),
        (0.15, "0.15"),
        ("1/3", "1/3"),
        ("2/6", "1/3"),
        ("2/3", "2/3"),
        ("1/7", "1/7"),
        ("1/6", "1/6"),
        ("3/7", "3/7"),
        ("9/10", "0.9"),
    ],
)
def test_canonical_token_strings(spelling, canonical):
    assert Token.parse(spelling).token == canonical


def test_a_third_stays_rational_and_records_float_and_hex():
    third = Token.parse("1/3")
    assert third.value == Fraction(1, 3)
    assert third.float64 == 1 / 3 and third.hex == (1 / 3).hex() == "0x1.5555555555555p-2"
    assert not third.is_exact  # no double equals 1/3
    assert Token.parse("0.1").hex == "0x1.999999999999ap-4"
    assert Token.parse("0.5").is_exact and Token.parse("0.25").is_exact and Token.parse(1).is_exact
    # the float passed to the simulator is not the exact rational: the token is the identity
    assert Fraction(Token.parse("0.1").float64) != Fraction(1, 10)


@pytest.mark.parametrize("value", [0.1, 0.2, 0.3, 0.7, 1e-05, 2.5e-07, 0.123456789, 0.999999])
def test_a_json_number_is_read_through_its_shortest_decimal(value):
    token = Token.parse(value)
    assert token.float64 == value  # the round trip returns the float that was written
    assert token.value == Fraction(repr(value))
    assert token == Token.parse(json.loads(json.dumps(value)))


@pytest.mark.parametrize(
    "bad, fragment",
    [
        ("", "decimal"),
        ("abc", "decimal"),
        ("0x1", "decimal"),
        ("1_0", "decimal"),
        ("1/0", "denominator 0"),
        ("-0.1", "[0, 1]"),
        ("1.5", "[0, 1]"),
        ("2", "[0, 1]"),
        ("3/2", "[0, 1]"),
        (-1, "[0, 1]"),
        (2, "[0, 1]"),
        (1.0000001, "[0, 1]"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (True, "boolean"),
        (None, "decimal string"),
        ([0.1], "decimal string"),
        (b"0.1", "decimal string"),
    ],
)
def test_malformed_tokens_are_refused_with_a_message_naming_the_field(bad, fragment):
    with pytest.raises(ValueError, match=fragment) as info:
        Token.parse(bad, what="manifest.p[2]")
    assert str(info.value).startswith("manifest.p[2]")


def test_token_constructor_guards_its_value():
    with pytest.raises(ValueError, match="Fraction in"):
        Token(Fraction(3, 2))
    with pytest.raises(ValueError, match="Fraction in"):
        Token(0.5)  # type: ignore[arg-type]


def test_tokens_order_by_value_and_hash_by_value():
    tokens = [Token.parse(x) for x in ("1/3", "0.1", "0.5", "1", "0", "2/3")]
    assert [t.token for t in sorted(tokens)] == ["0", "0.1", "1/3", "0.5", "2/3", "1"]
    assert len({Token.parse("0.5"), Token.parse("1/2"), Token.parse(0.5)}) == 1


@PROPERTY
@given(st.integers(0, 2000), st.integers(1, 2000))
def test_every_token_string_parses_back_to_itself_in_lowest_terms(num, den):
    if num > den:
        num, den = den, num
    token = Token.parse(f"{num}/{den}")
    assert token.value == Fraction(num, den)
    assert Token.parse(token.token) == token  # the string is an identifier
    text = token.token
    assert ("/" in text) == any(
        p not in (2, 5) for p in _prime_factors(token.value.denominator)
    )  # rational exactly when the decimal does not terminate
    if "/" in text:
        top, bottom = text.split("/")
        assert math.gcd(int(top), int(bottom)) == 1 and int(bottom) > 1
    else:  # an independent computation of the expansion: Decimal is exact here
        expected = Decimal(token.value.numerator) / Decimal(token.value.denominator)
        assert text == format(expected.normalize(), "f")
        assert not text.endswith(".") and ("." not in text or not text.endswith("0"))


def _prime_factors(k: int) -> set[int]:
    out, p = set(), 2
    while p * p <= k:
        while k % p == 0:
            out.add(p)
            k //= p
        p += 1
    if k > 1:
        out.add(k)
    return out


@PROPERTY
@given(st.integers(0, 10**9), st.integers(1, 10**9))
def test_float64_is_the_nearest_double_and_hex_round_trips(num, den):
    if num > den:
        num, den = den, num
    token = Token.parse(f"{num}/{den}")
    x = token.float64
    v = token.value
    # the neighbors on both sides are no closer to the exact value
    below, above = math.nextafter(x, -math.inf), math.nextafter(x, math.inf)
    err = abs(Fraction(x) - v)
    assert err <= abs(Fraction(below) - v) and err <= abs(Fraction(above) - v)
    assert float.fromhex(token.hex) == x


@PROPERTY
@given(st.integers(1, 15), st.data())
def test_a_decimal_of_up_to_15_digits_survives_the_trip_through_a_json_number(digits, data):
    # repr(float) is the shortest decimal that round-trips, and no two decimals of at most 15
    # significant digits share a double (DBL_DIG = 15): reading a JSON number gives the decimal
    # the author typed
    body = data.draw(st.text("0123456789", min_size=digits, max_size=digits))
    text = f"0.{body}"
    assert Token.parse(float(text)) == Token.parse(text)
    assert Token.parse(float(text)).token == Token.parse(text).token
    exponent = data.draw(st.integers(-8, 0))
    scaled = f"{body}e{exponent - digits}"  # the same digits at another scale
    assert Token.parse(float(scaled)) == Token.parse(scaled)


# ---------------------------------------------------------------------------
# the trajectory key
# ---------------------------------------------------------------------------


def test_trajectory_key_has_a_canonical_string_and_parses_back():
    key = TrajectoryKey("ws-pilot-001", 16, "0.1", 0.15, 3, 1)
    assert str(key) == "ws-pilot-001:n=16:beta=0.1:p=0.15:g=3:c=1:r=0"
    assert TrajectoryKey.parse(str(key)) == key
    third = TrajectoryKey("e", 24, 0, "1/3", 0, 0)
    assert str(third) == "e:n=24:beta=0:p=1/3:g=0:c=0:r=0"
    assert TrajectoryKey.parse(str(third)) == third
    assert key.stream_index(8) == 3 * 8 + 1


def test_trajectory_key_is_frozen_hashable_and_normalizes_its_tokens():
    a = TrajectoryKey("e", 16, "0.1", "1/10", 2, 0)
    b = TrajectoryKey("e", np.int64(16), 0.1, Fraction(1, 10), np.int64(2), 0)
    assert a == b and hash(a) == hash(b) and {a: 1}[b] == 1
    assert isinstance(a.n, int) and isinstance(b.n, int) and isinstance(b.g, int)
    with pytest.raises(AttributeError):
        a.n = 17  # type: ignore[misc]
    assert sorted([TrajectoryKey("e", 16, 0, "0.2", 0, 0), TrajectoryKey("e", 16, 0, "0.1", 1, 0)])[
        0
    ].p == Token.parse("0.1")


@pytest.mark.parametrize(
    "args, fragment",
    [
        (("", 16, 0, 0.1, 0, 0), "experiment_id"),
        (("a b", 16, 0, 0.1, 0, 0), "experiment_id"),
        (("-a", 16, 0, 0.1, 0, 0), "experiment_id"),
        (("a:b", 16, 0, 0.1, 0, 0), "experiment_id"),
        (("x" * 65, 16, 0, 0.1, 0, 0), "experiment_id"),
        (("e", 0, 0, 0.1, 0, 0), "n must be at least 1"),
        (("e", 16.0, 0, 0.1, 0, 0), "n must be an integer"),
        (("e", 16, 2, 0.1, 0, 0), "beta"),
        (("e", 16, 0, "x", 0, 0), "p must be"),
        (("e", 16, 0, 0.1, -1, 0), "g must be at least 0"),
        (("e", 16, 0, 0.1, 0, True), "c must be an integer"),
    ],
)
def test_trajectory_key_refuses_bad_fields(args, fragment):
    with pytest.raises(ValueError, match=fragment):
        TrajectoryKey(*args)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "e:n=16:beta=0:p=0.1:g=0:c=0",
        "e:n=16:beta=0:p=0.1:g=0:c=0:r=0:x=1",
        "e:m=16:beta=0:p=0.1:g=0:c=0:r=0",
        "e:n=:beta=0:p=0.1:g=0:c=0:r=0",
        "e:n=-1:beta=0:p=0.1:g=0:c=0:r=0",
        "e:n=sixteen:beta=0:p=0.1:g=0:c=0:r=0",
    ],
)
def test_trajectory_key_parse_refuses_malformed_text(text):
    with pytest.raises(ValueError, match="trajectory key"):
        TrajectoryKey.parse(text)


@PROPERTY
@given(
    st.from_regex(r"[A-Za-z0-9][A-Za-z0-9._-]{0,20}", fullmatch=True),
    st.integers(1, 5000),
    st.integers(0, 1000),
    st.integers(1, 1000),
    st.integers(0, 10**6),
    st.integers(0, 10**6),
    st.integers(0, 3),
)
def test_trajectory_key_round_trips(experiment, n, num, den, g, c, r):
    p = Fraction(min(num, den), den)
    key = TrajectoryKey(
        experiment, n, p, Fraction(num % (den + 1), den) if num <= den else 0, g, c, r
    )
    assert TrajectoryKey.parse(str(key)) == key


# ---------------------------------------------------------------------------
# seeds and the stream index
# ---------------------------------------------------------------------------


def test_graph_seed_is_the_master_seed_and_the_graph_label():
    assert graph_seed(17, 3) == (17, 3)
    assert graph_seed(0, 0) == (0, 0)
    assert graph_seed(2**32 - 1, 2**32 - 1) == (2**32 - 1, 2**32 - 1)
    assert graph_seed(np.int64(17), np.int64(3)) == (17, 3)
    for bad in [(-1, 0), (0, -1), (2**32, 0), (0, 2**32), (1.5, 0), (True, 0), ("1", 0)]:
        with pytest.raises(ValueError, match="master_graph_seed|g must|g:|integer"):
            graph_seed(*bad)  # type: ignore[arg-type]


def test_graph_seed_is_the_native_collection_convention():
    # "realizations": G builds graph i from the seed [seed, i] (api._build_graph, v1.4 section 7.2)
    spec = {"family": "watts_strogatz", "k": 4, "beta": 0.1, "seed": 17, "realizations": 4}
    for g, edges in enumerate(api._build_graph(spec, 16)):
        assert np.array_equal(edges, graphs.watts_strogatz(16, 4, 0.1, seed=graph_seed(17, g)))
        assert np.array_equal(edges, build_graph(16, "0.1", g, 17).edges)


def test_graph_seed_gives_the_stream_the_graph_generator_uses():
    # (master, g) and [master, g] are the same SeedSequence, and different pairs differ
    a = np.random.SeedSequence(graph_seed(17, 3)).generate_state(4)
    assert np.array_equal(a, np.random.SeedSequence([17, 3]).generate_state(4))
    pairs = [(m, g) for m in (0, 1, 17, 2**32 - 1) for g in (0, 1, 2, 2**32 - 1)]
    states = {tuple(np.random.SeedSequence(p).generate_state(4).tolist()) for p in pairs}
    assert len(states) == len(pairs)


def test_graph_zero_shares_its_seed_with_the_plain_master_seed():
    # numpy pads the entropy with zero words: (s, 0) is the SeedSequence of s itself. If the master
    # circuit seed were the master graph seed, graph 0 would be built from the very words that
    # key the circuit streams, which is why the manifest refuses equal master seeds.
    plain = np.random.SeedSequence(17).generate_state(4)
    assert np.array_equal(np.random.SeedSequence(graph_seed(17, 0)).generate_state(4), plain)
    assert not np.array_equal(np.random.SeedSequence(graph_seed(17, 1)).generate_state(4), plain)


def test_run_seed_shared_is_the_master_seed_for_every_rate():
    assert run_seed(29, "shared", "0.1") == 29
    assert run_seed(29, "shared", "1/3") == 29
    assert run_seed(0, "shared", 0.5) == 0


def test_run_seed_independent_hashes_the_rate_token():
    h = rate_hash("0.1")
    expected = int.from_bytes(hashlib.blake2b(b"0.1", digest_size=8).digest(), "little")
    assert h == expected == 15065691896991664069  # golden value
    assert run_seed(29, "independent", "0.1") == (29, h)
    assert run_seed(29, "independent", 0.1) == (29, h) == run_seed(29, "independent", "1/10")
    assert run_seed(29, "independent", "0.15") == (29, rate_hash("0.15"))
    rates = ["0", "0.05", "0.1", "0.15", "0.2", "1/3", "0.5", "1"]
    assert len({rate_hash(r) for r in rates}) == len(rates)
    # the first 8 bytes of a 64-byte BLAKE2b digest would be a different number
    assert h != int.from_bytes(hashlib.blake2b(b"0.1").digest()[:8], "little")


def test_run_seed_refuses_unknown_modes_and_bad_seeds():
    with pytest.raises(ValueError, match="gate_mode must be one of"):
        run_seed(29, "coupled", "0.1")
    with pytest.raises(ValueError, match="master_circuit_seed"):
        run_seed(-1, "shared", "0.1")
    with pytest.raises(ValueError, match="2\\*\\*32"):
        run_seed(2**32, "shared", "0.1")


def test_stream_index_values_and_inverse():
    assert stream_index(0, 0, 8) == 0
    assert stream_index(0, 7, 8) == 7
    assert stream_index(3, 1, 8) == 25
    assert stream_index(19, 1, 8) == 153
    assert split_stream_index(25, 8) == (3, 1)
    assert stream_index(np.int64(2), np.int64(1), np.int64(4)) == 9
    assert stream_index(5, 0, 1) == 5  # stride 1: one circuit per graph, u = g


def test_stream_index_refuses_a_circuit_beyond_the_stride():
    with pytest.raises(ValueError, match="below the circuit index stride 8"):
        stream_index(0, 8, 8)
    with pytest.raises(ValueError, match="frozen"):
        stream_index(2, 9, 8)
    for bad in [(-1, 0, 8), (0, -1, 8), (0, 0, 0), (0.0, 0, 8), (0, 0, 8.0)]:
        with pytest.raises(ValueError):
            stream_index(*bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="int64"):
        stream_index(2**63, 0, 2)


@PROPERTY
@given(st.integers(1, 64), st.integers(1, 40), st.data())
def test_stream_index_is_one_to_one_and_ignores_the_totals(stride, graphs_, data):
    # u depends on (g, c, stride) only: the same (g, c) has the same u however many graphs or
    # circuits the experiment has, so extending it never renumbers an existing trajectory.
    c_count = data.draw(st.integers(1, stride))
    seen = {}
    for g in range(graphs_):
        for c in range(c_count):
            u = stream_index(g, c, stride)
            assert u == g * stride + c and u // stride == g and u % stride == c
            assert split_stream_index(u, stride) == (g, c)
            assert u not in seen
            seen[u] = (g, c)
    more = data.draw(st.integers(0, 10))
    assert all(stream_index(g, c, stride) == u for u, (g, c) in seen.items())
    assert stream_index(graphs_ + more, 0, stride) not in seen  # a new graph, a new index


def test_the_engine_runs_trajectory_u_on_graph_u_over_stride():
    # circuits_per_graph = stride: trajectory u = g * stride + c uses graph g, whatever c < stride
    stride, n = 4, 9
    records = [build_graph(n, "0.3", g, 17) for g in range(3)]
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=[r.edges for r in records],
        circuits_per_graph=stride,
        layers=30,
        measurement_rate=0.1,
        schedule="random_edges",
        reference="purification",
    )
    assert spec.num_trajectories == 3 * stride
    for g in range(3):
        allowed = {tuple(e) for e in records[g].edges.tolist()}
        others = [{tuple(e) for e in r.edges.tolist()} for k, r in enumerate(records) if k != g]
        for c in (0, stride - 1):  # the unused circuit labels are valid too
            u = stream_index(g, c, stride)
            layers = wsref.parse_circuit(str(spec.circuit(u, seed=29)))
            used = {(min(a, b), max(a, b)) for layer in layers for a, b, _ in layer.gates}
            assert used <= allowed
            assert all(not used <= other for other in others)  # 150 gates reveal the graph


# ---------------------------------------------------------------------------
# graphs and records
# ---------------------------------------------------------------------------


def _explicit_circulant(n: int) -> set[tuple[int, int]]:
    out = set()
    for i in range(n):
        for d in (1, 2):
            j = (i + d) % n
            if i != j:
                out.add((min(i, j), max(i, j)))
    return out


@pytest.mark.parametrize("n", list(range(1, 14)))
def test_circulant_edges_follow_the_definition(n):
    edges = circulant_edges(n)
    assert {tuple(e) for e in edges.tolist()} == _explicit_circulant(n)
    assert len(edges) == n * min(4, n - 1) // 2  # |E_0(n)| of v1.4 section 2.1
    if n <= 5:  # K_n
        assert len(edges) == n * (n - 1) // 2
    else:
        assert len(edges) == 2 * n and set(graphs.degrees(edges, n).tolist()) == {4}
    if n >= 1 and len(edges):
        assert np.all(edges[:, 0] < edges[:, 1])
        assert edges.tolist() == sorted(edges.tolist())


@pytest.mark.parametrize("n", list(range(5, 41)))
def test_original_edge_list_is_offset_major_and_matches_the_native_ring(n):
    listed = original_edge_list(n)
    assert listed == wsref.original_edge_list(n)
    assert listed == graphs._ring_lattice(n, 4)  # the native generator's own list
    assert listed[:3] == [(0, 1), (1, 2), (2, 3)] and listed[n] == (0, 2)  # d = 1 first, then 2
    assert all(b == (a + 1) % n for a, b in listed[:n]) and all(
        b == (a + 2) % n for a, b in listed[n:]
    )
    # 2n pairwise distinct unordered pairs: the invariant behind the absent-edge abort
    assert len({frozenset(e) for e in listed}) == 2 * n
    assert {frozenset(e) for e in listed} == {frozenset(e) for e in circulant_edges(n).tolist()}


def test_original_edge_list_needs_n_at_least_five():
    with pytest.raises(ValueError, match="at least 5"):
        original_edge_list(4)


def _independent_digest(edges, n: int) -> str:
    payload = b"SGF2EDGES1\x00" + struct.pack("<I", n)
    for a, b in edges:
        payload += struct.pack("<ii", a, b)
    return hashlib.sha256(payload).hexdigest()


def test_build_graph_returns_the_validated_record_of_the_native_graph():
    record = build_graph(16, "0.1", 3, 17)
    assert isinstance(record, GraphRecord)
    assert (record.n, record.beta, record.g) == (16, Token.parse("0.1"), 3)
    assert record.seed == (17, 3) and record.edge_order == "offset-major"
    assert record.generator == ident.GRAPH_GENERATOR_VERSION
    native = graphs.watts_strogatz(16, 4, 0.1, seed=(17, 3))
    assert np.array_equal(record.edges, native) and record.num_edges == 32
    assert not record.edges.flags.writeable and record.edges.dtype == np.int64
    assert record.edge_digest == edge_digest(native, 16) == _independent_digest(native.tolist(), 16)
    record.verify()
    assert record.regenerate() == record
    # nothing but (n, beta, g, master seed) enters: rebuilding in any order gives the same record
    again = [build_graph(16, 0.1, g, 17) for g in (3, 2, 1, 0)]
    assert again[0] == record and len({r.edge_digest for r in again}) == 4


def test_the_digest_is_pinned_and_agrees_with_the_independent_reference():
    # golden vectors: the reference rewiring (_ws_reference, no package code) and the digest
    # computed with struct and hashlib in this file
    cases = {
        (16, "0.1", 3, 17): "ccc5c1aaedfb3941f462c6dbfe0a3ecdd0fa98106c89863539c5d51434abfe45",
        (24, "0", 0, 17): None,
        (24, "1", 5, 29): None,
    }
    for (n, beta, g, master), pinned in cases.items():
        record = build_graph(n, beta, g, master)
        reference = wsref.rewire(n, Token.parse(beta).float64, (master, g))
        assert np.array_equal(record.edges, reference)
        assert record.edge_digest == _independent_digest(reference.tolist(), n)
        if pinned is not None:
            assert record.edge_digest == pinned


def test_beta_zero_graph_is_the_circulant_and_n_five_is_complete():
    for n in (5, 6, 7, 10, 25):
        assert np.array_equal(build_graph(n, "0", 4, 17).edges, circulant_edges(n))
    complete = {(a, b) for a in range(5) for b in range(a + 1, 5)}
    for beta in ("0", "0.3", "1"):
        assert {tuple(e) for e in build_graph(5, beta, 2, 17).edges.tolist()} == complete


def test_graph_record_identity_comes_from_the_digest_and_the_provenance():
    a, b = build_graph(12, "0.2", 1, 17), build_graph(12, "0.2", 1, 17)
    assert a == b and hash(a) == hash(b)
    assert a != build_graph(12, "0.2", 2, 17) and a != build_graph(12, "0.3", 1, 17)
    assert a != build_graph(12, "0.2", 1, 18)
    assert a != "not a record"
    d = a.to_dict()
    assert d["n"] == 12 and d["beta"] == "0.2" and d["seed"] == [17, 1] and d["num_edges"] == 24
    assert "edges" not in d and a.to_dict(include_edges=True)["edges"] == a.edges.tolist()
    json.dumps(d)  # plain JSON


def test_verify_detects_a_tampered_graph():
    record = build_graph(12, "0.2", 1, 17)
    swapped = record.edges.copy()
    swapped[0] = swapped[1]  # a repeated edge
    bad = GraphRecord(**{**record.__dict__, "edges": swapped})
    with pytest.raises(ValueError, match="repeated"):
        bad.verify()
    other = build_graph(12, "0.2", 2, 17)  # a valid graph, but not this record's
    forged = GraphRecord(**{**record.__dict__, "edges": other.edges})
    with pytest.raises(ValueError, match="does not match its digest"):
        forged.verify()
    wrong_seed = GraphRecord(**{**record.__dict__, "seed": (17, 9)})
    with pytest.raises(ValueError, match="does not reproduce"):
        wrong_seed.regenerate()


def _bad_graph(kind: str, n: int = 8) -> np.ndarray:
    edges = circulant_edges(n).copy()
    if kind == "loop":
        edges[3] = (2, 2)
    elif kind == "unoriented":
        edges[3] = edges[3][::-1]
    elif kind == "duplicate":
        edges = np.vstack([edges[:-1], edges[-2:-1]])
    elif kind == "unsorted":
        edges[[2, 3]] = edges[[3, 2]]
    elif kind == "range":
        edges[-1] = (0, n)
    elif kind == "short":
        edges = edges[:-1]
    elif kind == "long":
        edges = np.unique(np.vstack([edges, [[0, 4]]]), axis=0)
    return edges


@pytest.mark.parametrize(
    "kind, fragment",
    [
        ("loop", "self-loop"),
        ("unoriented", "oriented"),
        ("duplicate", "repeated"),
        ("unsorted", "sorted"),
        ("range", "outside"),
        ("short", "expected 2n = 16 edges"),
        ("long", "expected 2n = 16 edges"),
    ],
)
def test_validate_graph_refuses_what_the_law_cannot_produce(kind, fragment):
    with pytest.raises(ValueError, match=fragment):
        validate_graph(_bad_graph(kind), 8, "0.1")


def test_validate_graph_refuses_low_degree_and_wrong_beta_zero_graphs():
    # rewire edge (0, 1) away from 0 and 1's partner so that vertex 1 keeps only one edge
    edges = {tuple(e) for e in circulant_edges(9).tolist()}
    for dropped in [(0, 1), (1, 2), (1, 3)]:
        edges.discard(dropped)
    edges |= {(4, 8), (5, 8), (2, 6)}
    graph = np.array(sorted(edges))
    assert len(graph) == 18 and graphs.degrees(graph, 9).min() < 2
    with pytest.raises(ValueError, match="degree below 2"):
        validate_graph(graph, 9, "0.5")
    rewired = graphs.watts_strogatz(12, 4, 1.0, seed=(17, 0))
    assert not np.array_equal(rewired, circulant_edges(12))
    validate_graph(rewired, 12, "0.5")
    with pytest.raises(ValueError, match="beta = 0 must give"):
        validate_graph(rewired, 12, "0")
    validate_graph(circulant_edges(12), 12, "0")


def test_validate_graph_refuses_non_integer_edges_and_small_n():
    with pytest.raises(ValueError, match="integer array"):
        validate_graph(np.zeros((10, 2)), 5, "0.1")
    with pytest.raises(ValueError, match="integer array"):
        validate_graph(np.zeros((10, 3), dtype=int), 5, "0.1")
    with pytest.raises(ValueError, match="at least 5"):
        validate_graph(circulant_edges(4), 4, "0.1")


def test_build_graph_refuses_unsupported_arguments():
    with pytest.raises(ValueError, match="at least 5"):
        build_graph(4, "0.1", 0, 17)
    with pytest.raises(ValueError, match="beta"):
        build_graph(8, "1.5", 0, 17)
    with pytest.raises(ValueError, match="g must"):
        build_graph(8, "0.1", -1, 17)
    with pytest.raises(ValueError, match="master_graph_seed"):
        build_graph(8, "0.1", 0, 2**32)


def test_edge_digest_is_strict_about_canonical_form():
    good = circulant_edges(7)
    assert edge_digest(good, 7) == _independent_digest(good.tolist(), 7)
    assert edge_digest(good, 7) != edge_digest(good, 8)  # n is part of the digest
    with pytest.raises(ValueError, match="oriented"):
        edge_digest(good[:, ::-1], 7)
    with pytest.raises(ValueError, match="sorted"):
        edge_digest(good[::-1], 7)
    with pytest.raises(ValueError, match="endpoints"):
        edge_digest(good, 6)
    with pytest.raises(ValueError, match="integer array"):
        edge_digest(good.astype(float), 7)


# ---------------------------------------------------------------------------
# canonical JSON and hashes
# ---------------------------------------------------------------------------


def test_canonical_json_sorts_keys_and_drops_whitespace():
    a = {"b": 1, "a": {"y": [1, 2.5, "x"], "x": None}, "c": True}
    b = {"c": True, "a": {"x": None, "y": [1, 2.5, "x"]}, "b": 1}
    assert (
        canonical_json(a) == canonical_json(b) == '{"a":{"x":null,"y":[1,2.5,"x"]},"b":1,"c":true}'
    )
    assert json.loads(canonical_json(a)) == a
    assert json_digest(a) == json_digest(b) == sha256_hex(canonical_json(a).encode())


def test_canonical_json_writes_tokens_as_strings_never_floats():
    doc = {"p": [Token.parse("0.1"), Token.parse("1/3")], "beta": Token.parse(0)}
    assert canonical_json(doc) == '{"beta":"0","p":["0.1","1/3"]}'
    with pytest.raises(ValueError, match="Token"):
        canonical_json({"p": Fraction(1, 10)})


def test_canonical_json_normalizes_numpy_tuples_and_negative_zero():
    doc = {
        "a": np.int64(3),
        "b": np.float64(0.5),
        "c": (1, 2),
        "d": np.array([[1, 2], [3, 4]]),
        "e": -0.0,
        "f": np.bool_(True),
    }
    assert canonical_json(doc) == '{"a":3,"b":0.5,"c":[1,2],"d":[[1,2],[3,4]],"e":0.0,"f":true}'
    assert canonical_json("é") == '"\\u00e9"'  # ASCII only


@pytest.mark.parametrize(
    "bad, fragment",
    [
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        ({"a": float("-inf")}, r"\$\.a"),
        ({1: 2}, "keys must be strings"),
        (b"x", "bytes"),
        ({"x": [1, {"y": object()}]}, r"\$\.x\[1\]\.y"),
        ({1, 2}, "set"),
    ],
)
def test_canonical_json_refuses_what_json_cannot_hold(bad, fragment):
    with pytest.raises(ValueError, match=fragment):
        canonical_json(bad)


def test_sha256_hex_matches_the_published_vectors():
    assert sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert sha256_hex(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# ---------------------------------------------------------------------------
# versions and identifiers
# ---------------------------------------------------------------------------


def test_version_record_names_every_version_and_the_native_contract():
    record = ident.version_record()
    assert record["randomness_contract"] == mipt.RANDOMNESS_CONTRACT == 2
    assert record["graph_generator"] == ident.GRAPH_GENERATOR_VERSION
    assert record["clifford_enumeration"] == ident.CLIFFORD_ENUMERATION_VERSION
    assert record["dynamics"] == ident.DYNAMICS_VERSION
    assert record["rowspace_format"] == ident.ROWSPACE_FORMAT_VERSION == "SGF2RS1"
    assert record["store_schema"] == ident.STORE_SCHEMA_VERSION
    assert all(isinstance(v, (str, int)) and v != "" for v in record.values())
    json.dumps(record)


@pytest.mark.parametrize("good", ["a", "ws-pilot-001", "A.b_c-9", "0", "x" * 64])
def test_experiment_ids_accept_file_safe_names(good):
    assert ident.check_experiment_id(good) == good


@pytest.mark.parametrize("bad", ["", " a", "a b", "a/b", "a:b", ".a", "_a", "é", "x" * 65, None, 3])
def test_experiment_ids_refuse_unsafe_names(bad):
    with pytest.raises(ValueError, match="experiment_id"):
        ident.check_experiment_id(bad)
