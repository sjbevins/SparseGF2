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
import itertools
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
from sparsegf2 import graphs, mipt
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


# Golden words, computed from hashlib and struct and pinned: the 64-bit BLAKE2b hash of the
# canonical token (digest_size=8, little-endian) as its low and high 32-bit words.
WORDS = {
    "0": (2642226052, 1977601347),
    "0.1": (2007713733, 3507754741),
    "0.15": (3479488031, 1099992840),
    "1/3": (3963472482, 2705452238),
}
GRAPH_TAG, CIRCUIT_TAG = 0x57534731, 0x57534331  # b"WSG1" and b"WSC1" as big-endian integers
RATES = ["0", "0.05", "0.1", "0.15", "0.2", "1/3", "0.5", "1"]


def _state(seed) -> tuple[int, ...]:
    """The four state words the generators read from a seed: ``SeedSequence(seed).generate_state(4)``."""
    return tuple(np.random.SeedSequence(seed).generate_state(4).tolist())


def test_the_seed_tags_spell_their_domains():
    assert ident.GRAPH_SEED_TAG == GRAPH_TAG == int.from_bytes(b"WSG1", "big") == wsref.GRAPH_TAG
    assert ident.CIRCUIT_SEED_TAG == CIRCUIT_TAG == int.from_bytes(b"WSC1", "big")
    assert ident.CIRCUIT_SEED_TAG == wsref.CIRCUIT_TAG
    assert ident.GRAPH_SEED_TAG != ident.CIRCUIT_SEED_TAG
    assert max(ident.GRAPH_SEED_TAG, ident.CIRCUIT_SEED_TAG) < 2**32  # one word each
    assert (ident.GRAPH_SEED_TAG, ident.CIRCUIT_SEED_TAG) == (1465075505, 1465074481)


@pytest.mark.parametrize("text, words", list(WORDS.items()))
def test_token_words_are_the_low_and_high_halves_of_the_blake2b_hash(text, words):
    digest = hashlib.blake2b(text.encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    assert (
        words == (value % 2**32, value // 2**32) == struct.unpack("<II", digest)
    )  # the golden words
    assert ident.token_words(text) == words == wsref.token_words(text)
    assert words[0] + (words[1] << 32) == rate_hash(text) == value
    assert all(0 <= w < 2**32 for w in words)
    # the digest of 8 bytes, not the first 8 bytes of the 64-byte digest
    assert words != struct.unpack("<II", hashlib.blake2b(text.encode()).digest()[:8])


def test_token_words_name_the_token_and_not_its_spelling():
    assert (
        ident.token_words(0.1)
        == ident.token_words("1/10")
        == ident.token_words(" .10 ")
        == WORDS["0.1"]
    )
    assert ident.token_words(Token.parse("3/20")) == ident.token_words(0.15) == WORDS["0.15"]
    assert ident.token_words(Fraction(1, 3)) == WORDS["1/3"]
    words = [ident.token_words(r) for r in RATES]
    assert len(set(words)) == len(RATES)  # different tokens, different words
    with pytest.raises(ValueError, match="must lie in"):
        ident.token_words("1.5")


def test_graph_seed_is_the_tagged_tuple_of_master_size_beta_and_graph():
    seed = graph_seed(17, 16, "0.1", 3)
    assert seed == (
        0x57534731,
        17,
        16,
        2007713733,
        3507754741,
        3,
    )  # spelled out with the golden words
    assert seed == (GRAPH_TAG, 17, 16, *WORDS["0.1"], 3) == wsref.graph_seed(17, 16, "0.1", 3)
    assert len(seed) == 6 and all(type(w) is int and 0 <= w < 2**32 for w in seed)
    # the token, not its spelling, names beta; numpy integers are accepted
    assert graph_seed(17, 16, 0.1, 3) == graph_seed(17, 16, "1/10", 3) == seed
    assert graph_seed(17, 16, Token.parse("0.10"), 3) == seed
    assert graph_seed(np.int64(17), np.int64(16), "0.1", np.int64(3)) == seed
    # the largest words are single words
    top = 2**32 - 1
    assert graph_seed(top, top, "1/3", top) == (GRAPH_TAG, top, top, *WORDS["1/3"], top)
    # the master graph seed 0 and graph label 0 give a seed with zero words, still six words long
    assert graph_seed(0, 5, "0", 0) == (GRAPH_TAG, 0, 5, *WORDS["0"], 0)


@pytest.mark.parametrize(
    "bad, fragment",
    [
        ((-1, 16, "0.1", 0), "master_graph_seed must be at least 0"),
        ((2**32, 16, "0.1", 0), r"master_graph_seed must be below 2\*\*32"),
        ((1.5, 16, "0.1", 0), "master_graph_seed must be an integer"),
        ((True, 16, "0.1", 0), "master_graph_seed must be an integer"),
        ((17, -1, "0.1", 0), "n must be at least 0"),
        ((17, 2**32, "0.1", 0), r"n must be below 2\*\*32"),
        ((17, 16.0, "0.1", 0), "n must be an integer"),
        ((17, 16, "1.5", 0), "beta must lie in"),
        ((17, 16, "x", 0), "beta must be a decimal"),
        ((17, 16, "0.1", -1), "g must be at least 0"),
        ((17, 16, "0.1", 2**32), r"g must be below 2\*\*32"),
        ((17, 16, "0.1", "3"), "g must be an integer"),
    ],
)
def test_graph_seed_refuses_what_is_not_a_seed_word_or_a_token(bad, fragment):
    with pytest.raises(ValueError, match=fragment):
        graph_seed(*bad)


def test_graph_seeds_differ_across_masters_sizes_betas_and_labels():
    # (master, n, beta, g) is read back from the tuple, so equal seeds mean equal arguments; the
    # SeedSequence states are different too (a property of the hash behind it, checked on this
    # grid and not proved)
    grid = list(
        itertools.product(
            (0, 1, 17), (5, 6, 16, 17), ("0", "0.05", "0.1", "1/3"), (0, 1, 2, 2**32 - 1)
        )
    )
    seeds = [graph_seed(*args) for args in grid]
    assert len(set(seeds)) == len(grid)
    assert len({_state(s) for s in seeds}) == len(grid)
    for args, seed in zip(grid, seeds, strict=True):
        assert seed == wsref.graph_seed(*args)  # read from the definition in the reference
    # one coordinate at a time, from the study's example: the master, the size, beta (two words), g
    base = graph_seed(17, 16, "0.1", 3)
    for other, changed_words in [
        (graph_seed(18, 16, "0.1", 3), 1),
        (graph_seed(17, 17, "0.1", 3), 1),
        (graph_seed(17, 16, "0.2", 3), 2),
        (graph_seed(17, 16, "0.1", 4), 1),
    ]:
        assert sum(a != b for a, b in zip(base, other, strict=True)) == changed_words
        assert _state(base) != _state(other)


@PROPERTY
@given(
    st.tuples(
        st.integers(0, 2**32 - 1),
        st.integers(5, 2**32 - 1),
        st.fractions(min_value=0, max_value=1, max_denominator=60),
        st.integers(0, 2**32 - 1),
    ),
    st.integers(0, 4),
    st.data(),
)
def test_graph_seed_is_injective_in_the_condition_and_the_graph(args, which, data):
    master, n, beta, g = args
    other = [master, n, beta, g]
    if which < 4:  # change one coordinate; which == 4 keeps all of them
        other[which] = data.draw(
            [
                st.integers(0, 2**32 - 1),
                st.integers(5, 2**32 - 1),
                st.fractions(min_value=0, max_value=1, max_denominator=60),
                st.integers(0, 2**32 - 1),
            ][which]
        )
    same = (master, n, Token.parse(beta), g) == (
        other[0],
        other[1],
        Token.parse(other[2]),
        other[3],
    )
    assert (graph_seed(master, n, beta, g) == graph_seed(*other)) == same


def test_graph_seeds_and_run_seeds_never_coincide_even_for_equal_master_seeds():
    # ruling R-SEED: the tags make the two domains disjoint, so one master seed may serve both.
    # Before it, graph 0 of master s was SeedSequence([s, 0]), which numpy pads to SeedSequence(s):
    # the very words that key the circuit streams when the circuit seed was s.
    master = 17
    sizes, betas = (5, 16, 24), ("0", "0.1", "0.3", "1/3")
    graph_seeds = [graph_seed(master, n, b, g) for n in sizes for b in betas for g in range(4)]
    run_seeds = [
        run_seed(master, n, b, mode, p)
        for n in sizes
        for b in betas
        for mode, p in [("shared", None), ("independent", "0.1"), ("independent", "0.15")]
    ]
    assert len(set(graph_seeds)) == len(graph_seeds) == 48
    assert len(set(run_seeds)) == len(run_seeds) == 36
    assert not set(graph_seeds) & set(run_seeds)
    assert {s[0] for s in graph_seeds} == {GRAPH_TAG} and {s[0] for s in run_seeds} == {CIRCUIT_TAG}
    assert not {_state(s) for s in graph_seeds} & {_state(s) for s in run_seeds}
    plain = _state(master)  # the old collision: the plain seed s is the key of the circuit streams
    assert plain not in {_state(s) for s in graph_seeds + run_seeds}
    assert _state((master, 0)) == plain  # numpy's zero padding, the reason the old rule was needed
    # and the tag is what separates them: the same words under the other tag give another state
    g = graph_seed(master, 16, "0.1", 3)
    assert _state(g) != _state((CIRCUIT_TAG, *g[1:]))


def test_seedsequence_pads_short_entropy_with_zeros_but_not_beyond_its_pool_of_four_words():
    # the facts about numpy that the proof in the documentation uses
    assert _state([17]) == _state([17, 0]) == _state([17, 0, 0, 0]) == _state(17)
    assert _state([17, 5, 9]) == _state([17, 5, 9, 0])
    assert _state([17, 5, 9, 1]) != _state(
        [17, 5, 9, 1, 0]
    )  # the fifth word is mixed in, zero or not
    assert _state([17, 5, 9, 1, 7]) != _state([17, 5, 9, 1, 7, 0])
    assert _state([17, 5, 9, 1, 7]) != _state([17, 5, 9, 1])
    assert _state([17, 5, 9, 1, 0, 0]) != _state([17, 5, 9, 1, 0])
    # the first four words are hashed by position, so swapping two of them changes the state
    assert _state([17, 5, 9, 1, 7]) != _state([5, 17, 9, 1, 7])
    # the study's seeds: every length exceeds four, so trailing zero words (g = 0) are not dropped
    seed = graph_seed(17, 16, "0.1", 0)
    assert seed[-1] == 0 and _state(seed) != _state(seed[:-1])
    zero_master = graph_seed(
        0, 16, "0.1", 0
    )  # a master seed of 0 and graph 0: zero words inside and at the end
    assert len({_state(zero_master), _state(zero_master[:-1]), _state(zero_master[:2])}) == 3


def test_seed_tuples_have_one_fixed_length_per_domain_and_it_exceeds_the_pool():
    sizes, betas = (5, 16), ("0", "0.1")
    graph = {len(graph_seed(17, n, b, g)) for n in sizes for b in betas for g in (0, 9)}
    shared = {
        len(run_seed(29, n, b, "shared", p)) for n in sizes for b in betas for p in (None, "0.1")
    }
    independent = {
        len(run_seed(29, n, b, "independent", p)) for n in sizes for b in betas for p in RATES
    }
    assert (graph, shared, independent) == ({6}, {5}, {7})
    # SeedSequence hashes the first pool_size words by position (zero-padding a shorter entropy)
    # and mixes every further word in whole: numpy's pool, read from numpy, is what every study
    # seed must exceed for the padding argument of the documentation
    pool = np.random.SeedSequence(0).pool_size
    assert pool == 4 and min(graph | shared | independent) > pool


def test_run_seed_shared_is_one_tuple_per_condition_for_every_rate():
    seed = run_seed(29, 16, "0.1", "shared")
    assert seed == (0x57534331, 29, 16, 2007713733, 3507754741)  # spelled out with the golden words
    assert seed == (CIRCUIT_TAG, 29, 16, *WORDS["0.1"]) == wsref.run_seed(29, 16, "0.1")
    assert len(seed) == 5 and all(type(w) is int and 0 <= w < 2**32 for w in seed)
    for p in RATES + [0.5, Token.parse("0.2"), Fraction(3, 20), None]:
        assert (
            run_seed(29, 16, "0.1", "shared", p) == seed
        )  # the rate does not enter in shared mode
    assert run_seed(29, 16, 0.1, "shared") == run_seed(29, 16, "1/10", "shared") == seed
    assert run_seed(np.int64(29), np.int64(16), "0.1", "shared") == seed
    with pytest.raises(
        ValueError, match="p must lie in"
    ):  # a malformed rate is refused in every mode
        run_seed(29, 16, "0.1", "shared", "2")


def test_run_seed_independent_appends_the_words_of_the_rate():
    seed = run_seed(29, 16, "0.1", "independent", "0.15")
    assert seed == (0x57534331, 29, 16, 2007713733, 3507754741, 3479488031, 1099992840)
    assert seed == wsref.run_seed(29, 16, "0.1", "0.15") and len(seed) == 7
    assert seed[:5] == run_seed(29, 16, "0.1", "shared")
    assert (
        run_seed(29, 16, "0.1", "independent", 0.15)
        == run_seed(29, 16, "0.1", "independent", "3/20")
        == seed
    )
    assert seed[5] + (seed[6] << 32) == rate_hash(
        "0.15"
    )  # the words are the halves of the rate hash
    assert rate_hash("0.1") == 15065691896991664069  # golden value
    seeds = {run_seed(29, 16, "0.1", "independent", p) for p in RATES}
    assert len(seeds) == len(RATES)  # one seed per rate
    assert len({_state(s) for s in seeds}) == len(RATES)
    with pytest.raises(ValueError, match="needs p_token"):
        run_seed(29, 16, "0.1", "independent")


def test_run_seeds_differ_across_masters_sizes_and_betas_in_both_modes():
    for mode, p in [("shared", None), ("independent", "0.15")]:
        grid = list(
            itertools.product((0, 29, 2**32 - 1), (5, 6, 16, 17), ("0", "0.05", "0.1", "1/3"))
        )
        seeds = [run_seed(m, n, b, mode, p) for m, n, b in grid]
        assert len(set(seeds)) == len(grid)
        assert len({_state(s) for s in seeds}) == len(grid)
        for (m, n, b), seed in zip(grid, seeds, strict=True):
            assert seed == wsref.run_seed(m, n, b, p)
    # the coupling is across the rate only: p enters nothing but the independent tuple
    assert len({run_seed(29, 16, "0.1", "shared", p) for p in RATES}) == 1
    assert len({run_seed(29, 16, "0.1", "independent", p) for p in RATES}) == len(RATES)


def test_run_seed_refuses_unknown_modes_and_bad_arguments():
    with pytest.raises(ValueError, match="gate_mode must be one of"):
        run_seed(29, 16, "0.1", "coupled", "0.1")
    with pytest.raises(ValueError, match="master_circuit_seed"):
        run_seed(-1, 16, "0.1", "shared")
    with pytest.raises(ValueError, match="2\\*\\*32"):
        run_seed(2**32, 16, "0.1", "shared")
    with pytest.raises(ValueError, match="n must be at least 0"):
        run_seed(29, -1, "0.1", "shared")
    with pytest.raises(ValueError, match="beta must lie in"):
        run_seed(29, 16, "2", "shared")
    with pytest.raises(ValueError, match="p must lie in"):
        run_seed(29, 16, "0.1", "independent", "2")


def test_graph_seed_gives_the_stream_the_graph_generator_uses(monkeypatch):
    # The six words are the entropy as given. The native generator derives its xoshiro256** state
    # with _kernel.rng.seed_state; the reference rewiring derives it with its own seed_words. Both
    # are called here on the study's seeds, and the generator is watched while it builds a graph.
    from sparsegf2._kernel import rng as native

    for g in range(4):
        seed = graph_seed(17, 16, "0.1", g)
        assert seed == wsref.graph_seed(17, 16, "0.1", g)
        assert native.seed_state(seed).tolist() == wsref.seed_words(seed)
        assert native.seed_state(list(seed)).tolist() == wsref.seed_words(seed)
    seen = []

    def watched(seed):
        state = native.seed_state(seed)
        seen.append((seed, state.copy()))
        return state

    monkeypatch.setattr(graphs, "seed_state", watched)
    record = build_graph(16, "0.1", 3, 17)
    assert len(seen) == 1 and seen[0][0] == record.seed == graph_seed(17, 16, "0.1", 3)
    assert seen[0][1].tolist() == wsref.seed_words(record.seed)
    assert np.array_equal(record.edges, wsref.rewire(16, 0.1, record.seed))


def test_the_study_builds_its_graphs_with_the_native_generator_and_the_study_seed():
    # ruling R-SEED: graphs come from graphs.watts_strogatz seeded with the six-word tuple, not from
    # the native collection convention [seed, i] (api._build_graph), which shares seeds across sizes
    # and rewiring probabilities
    for g in range(4):
        record = build_graph(16, "0.1", g, 17)
        seed = (0x57534731, 17, 16, 2007713733, 3507754741, g)  # spelled out with the golden words
        assert record.seed == seed == graph_seed(17, 16, "0.1", g)
        assert np.array_equal(record.edges, graphs.watts_strogatz(16, 4, 0.1, seed=seed))
        assert np.array_equal(
            record.edges, wsref.rewire(16, 0.1, wsref.graph_seed(17, 16, "0.1", g))
        )
    old = [graphs.watts_strogatz(16, 4, 0.1, seed=(17, g)) for g in range(8)]
    new = [build_graph(16, "0.1", g, 17).edges for g in range(8)]
    assert (
        sum(np.array_equal(a, b) for a, b in zip(old, new, strict=True)) <= 1
    )  # another convention


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
    seed = run_seed(29, n, "0.3", "shared")  # the study's own run seed for the condition (9, 0.3)
    for g in range(3):
        allowed = {tuple(e) for e in records[g].edges.tolist()}
        others = [{tuple(e) for e in r.edges.tolist()} for k, r in enumerate(records) if k != g]
        for c in (0, stride - 1):  # the unused circuit labels are valid too
            u = stream_index(g, c, stride)
            layers = wsref.parse_circuit(str(spec.circuit(u, seed=seed)))
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
    assert (
        record.seed == (GRAPH_TAG, 17, 16, *WORDS["0.1"], 3) and record.edge_order == "offset-major"
    )
    assert record.generator == ident.GRAPH_GENERATOR_VERSION
    native = graphs.watts_strogatz(16, 4, 0.1, seed=record.seed)
    assert np.array_equal(record.edges, native) and record.num_edges == 32
    assert not record.edges.flags.writeable and record.edges.dtype == np.int64
    assert record.edge_digest == edge_digest(native, 16) == _independent_digest(native.tolist(), 16)
    record.verify()
    assert record.regenerate() == record
    # nothing but (n, beta, g, master seed) enters: rebuilding in any order gives the same record
    again = [build_graph(16, 0.1, g, 17) for g in (3, 2, 1, 0)]
    assert again[0] == record and len({r.edge_digest for r in again}) == 4


def test_the_digest_is_pinned_and_agrees_with_the_independent_reference():
    # golden vectors: the reference rewiring (_ws_reference, no package code) seeded with the
    # reference's own seed tuple, and the digest computed with struct and hashlib in this file
    cases = {
        (16, "0.1", 3, 17): "23df2ca38f4a3c47ecb35f795677530d515588dabffd16abbadfbeee77c43e6c",
        (16, "0.1", 0, 17): "584305b4853745a796fe4c6641d5e9c15cb6fe3f1118157ad2b98c1908215aa6",
        (24, "0", 0, 17): "02a2e2d6c52195fe0b83dbc2ab55d1cbb711c2eaa7824be4d7b14ccb241c5dc0",
        (24, "1", 5, 29): "c6f7d572b2f71e849a31674176a6db3bba4e3aad37897564453ac5c5c828e040",
        (12, "0.2", 1, 17): "0b3ddb56ad6b698a0510cafbbeb23aa6b67dc0d1ca6c1ec1ac753d214857139f",
        (24, "0.3", 7, 5): "1e0e8b4a9ffa0f812506baaf04bdb0145fa7885b6cebadc577187f37acfbaaff",
    }
    for (n, beta, g, master), pinned in cases.items():
        record = build_graph(n, beta, g, master)
        seed = wsref.graph_seed(master, n, beta, g)
        reference = wsref.rewire(n, Token.parse(beta).float64, seed)
        assert record.seed == seed
        assert np.array_equal(record.edges, reference)
        assert record.edge_digest == _independent_digest(reference.tolist(), n) == pinned
    # beta = 0 is the circulant graph whatever the seed: the digest does not depend on it
    assert build_graph(24, "0", 0, 17).edge_digest == build_graph(24, "0", 9, 3).edge_digest


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
    assert a != build_graph(13, "0.2", 1, 17)  # the same label g of another size is another graph
    assert a != "not a record"
    d = a.to_dict()
    assert d["n"] == 12 and d["beta"] == "0.2" and d["num_edges"] == 24
    assert d["seed"] == [GRAPH_TAG, 17, 12, *wsref.token_words("0.2"), 1] == list(a.seed)
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
    # the seed tuple is part of the record: any altered word, or a tuple of another length, fails
    master, tag = record.seed[1], record.seed[0]
    altered = [
        (tag, master + 1, *record.seed[2:]),  # another master seed
        (tag, master, record.seed[2] + 1, *record.seed[3:]),  # another size word
        (tag, master, *record.seed[2:5], record.seed[5] + 8),  # another graph word
        (CIRCUIT_TAG, *record.seed[1:]),  # the other domain's tag
        record.seed[:5],  # a tuple of five words
        (17, 1),  # the old convention
    ]
    for seed in altered:
        wrong_seed = GraphRecord(**{**record.__dict__, "seed": seed})
        with pytest.raises(ValueError, match="does not reproduce"):
            wrong_seed.regenerate()
    assert GraphRecord(**{**record.__dict__}).regenerate() == record  # the untouched record does


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
    # the seed derivation is versioned by its domain tags: a different derivation needs new tags
    assert record["graph_seed_tag"] == ident.GRAPH_SEED_TAG == 0x57534731
    assert record["circuit_seed_tag"] == ident.CIRCUIT_SEED_TAG == 0x57534331
    assert all(isinstance(v, (str, int)) and v != "" for v in record.values())
    json.dumps(record)


@pytest.mark.parametrize("good", ["a", "ws-pilot-001", "A.b_c-9", "0", "x" * 64])
def test_experiment_ids_accept_file_safe_names(good):
    assert ident.check_experiment_id(good) == good


@pytest.mark.parametrize("bad", ["", " a", "a b", "a/b", "a:b", ".a", "_a", "é", "x" * 65, None, 3])
def test_experiment_ids_refuse_unsafe_names(bad):
    with pytest.raises(ValueError, match="experiment_id"):
        ident.check_experiment_id(bad)
