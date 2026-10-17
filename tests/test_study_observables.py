"""Tests of sparsegf2.study.observables (specification v1.4 §5.2-§5.8; validation requirements 12, 13).

The references are independent of the production code: ``tests/_study_oracle.py`` enumerates
whole stabilizer groups, every Pauli commuting with a code, every subset, every permutation, every
cyclic interval and every connected subset, with Python ints and Fractions; the hand cases are
written out here. Engine runs supply real codes (several n, odd n, beta, p and depths); Stim
supplies random ones. Every statistical test states its tolerance and its false-alarm probability;
seeds are fixed.
"""

import dataclasses
import hashlib
import json
import math
import re
import string
import struct
import tracemalloc
from fractions import Fraction

import numpy as np
import pytest

import _study_oracle as oracle
from sparsegf2 import StabilizerCode, erasures, graphs, mipt
from sparsegf2.study import identity
from sparsegf2.study import observables as ob
from sparsegf2.study.identity import TrajectoryKey
from sparsegf2.study.manifest import CorrelatedErasure
from sparsegf2.tableau import PhaseFreeTableau

# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _edges(n, beta, g):
    """The study's graph g at (n, beta): Watts-Strogatz for n >= 5, the complete graph below."""
    if n < 5:
        return graphs.complete(n)
    return graphs.watts_strogatz(n, 4, beta, seed=(17, g))


def _run(n, beta, p, layers, probes, *, graphs_count=2, circuits=2, stop=False, seed=29):
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=[_edges(n, beta, g) for g in range(graphs_count)],
        circuits_per_graph=circuits,
        layers=layers,
        measurement_rate=p,
        reference="purification",
        schedule="random_edges",
        gates_per_layer=n // 2,
        random_orientation=False,
        periodic=True,
        checkpoints="every_layer",
        probes=[mipt.code_dimension(), *probes],
        stop_when_purified=stop,
    )
    return spec.run(seed=seed)


def _joint_oracle(result, row, evaluation, name="tableau_snapshot"):
    """The oracle's joint state of a stored snapshot, read from the raw engine words."""
    words, counts = result.snapshots[name]
    big = result.spec.num_qubits
    rows = oracle.words_to_ints(words[row, evaluation, : counts[row, evaluation]], big)
    return oracle.Joint(rows, result.spec.num_system)


def _qubits(mask, n):
    return [q for q in range(n) if (mask >> q) & 1]


def _binomial_p_value(x, trials, p):
    """Exact two-sided p-value of x successes in Bin(trials, p): the mass of outcomes no likelier."""
    logs = [
        math.lgamma(trials + 1)
        - math.lgamma(i + 1)
        - math.lgamma(trials - i + 1)
        + i * math.log(p)
        + (trials - i) * math.log1p(-p)
        for i in range(trials + 1)
    ]
    pmf = [math.exp(v) for v in logs]
    cut = pmf[x] * (1 + 1e-9)
    return min(1.0, sum(q for q in pmf if q <= cut))


# ---------------------------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------------------------

EXPECTED_IDS = {
    "k",
    "code_rate",
    "stabilizer_rank",
    "k_trace",
    "first_passage",
    "d_cyc",
    "uniform_failure",
    "D",
    "Phi",
    "correlated_failure",
    "D_J",
    "Phi_J",
    "mean_leakage",
    "d_conn",
    "half_cut_entropy",
    "peak_d_cyc",
    "peak_D",
    "peak_D_J",
    "peak_d_conn",
}
RETROSPECTIVE = {
    "k",
    "code_rate",
    "stabilizer_rank",
    "d_cyc",
    "mean_leakage",
    "d_conn",
    "half_cut_entropy",
}


def test_the_catalog_has_one_entry_per_observable():
    rows = ob.catalog()
    assert {r["id"] for r in rows} == EXPECTED_IDS
    assert len(rows) == len(EXPECTED_IDS)
    assert [(r["id"], r["version"]) for r in rows] == sorted((r["id"], r["version"]) for r in rows)
    for r in rows:
        assert r["level"] in ob.LEVELS
        assert "v1.4 §" in r["definition"], r["id"]
        assert r["function"] in ob.__all__
        for kind in r["inputs"]:
            if kind.startswith("trace:"):
                assert kind[len("trace:") :] in EXPECTED_IDS
            else:
                assert kind in ob.INPUTS
        json.dumps(r)  # plain Python values
    assert {r["id"] for r in rows if r["retrospective"]} == RETROSPECTIVE
    table = ob.catalog_markdown().splitlines()
    assert len(table) == len(rows) + 2 and table[0].startswith("| id |")
    # every row has the header's six cells, and each cell reads back as its text: the ids are
    # code spans, the definitions plain text (they hold _, *, [, <, ~ and |)
    for line, row in zip(table[2:], rows, strict=True):
        cells = _gfm_row(line)
        assert len(cells) == 6, line
        assert cells[0] == ("code", row["id"])
        assert cells[5] == ("text", row["definition"])
    assert sum("|" in r["definition"] for r in rows) >= 4  # the case that broke the table
    assert sum("_" in r["definition"] for r in rows) >= 10  # and the one that is inline syntax


_INLINE_OPENERS = "`*_[<&~$"
"""The characters that can begin inline syntax in a GitHub table cell, from the specifications:
code spans (`), emphasis and strong emphasis (* and _), links and images ([; an image is ![),
autolinks and raw HTML (<) and entity and numeric character references (&) (CommonMark 0.31
§6), strikethrough (~, GitHub Flavored Markdown) and math ($, GitHub). A "]" or "!" begins
nothing without a "[" and a backslash is handled as an escape."""


def _split_row(line, rule):
    """The cells of a table row before inline parsing. A "|" is a cell delimiter unless it is
    escaped: under rule "gfm" when a backslash precedes it (the GFM table extension, cmark-gfm's
    scanner), under rule "parity" when an odd run of backslashes precedes it (marked's
    splitCells). Either way the backslash just before an escaped "|" is dropped and the spaces
    and tabs at each cell's ends are removed."""
    cells, current, run = [], "", 0
    for ch in line.strip()[1:-1] + "|":
        escaped = run > 0 if rule == "gfm" else run % 2 == 1
        if ch == "|" and not escaped:
            cells.append(current.strip(" \t"))
            current = ""
        elif ch == "|":
            current = current[:-1] + "|"
        else:
            current += ch
        run = run + 1 if ch == "\\" else 0
    return cells


def _inline(cell):
    """(kind, shown text) of one cell's inline Markdown. A cell that is one code span shows its
    content literally ("code"; one space is stripped from each end when both ends have one).
    Otherwise a backslash before ASCII punctuation shows that character and any other character
    shows itself ("text"); an unescaped character of _INLINE_OPENERS could begin Markdown
    syntax, so the cell is then reported as ("markdown", cell): this model does not vouch for
    what it shows."""
    if len(cell) >= 2 and cell[0] == cell[-1] == "`" and "`" not in cell[1:-1]:
        inner = cell[1:-1]
        if len(inner) >= 2 and inner[0] == inner[-1] == " " and inner.strip(" "):
            inner = inner[1:-1]
        return ("code", inner)
    text, i = "", 0
    while i < len(cell):
        if cell[i] == "\\" and i + 1 < len(cell) and cell[i + 1] in string.punctuation:
            text += cell[i + 1]
            i += 2
        elif cell[i] in _INLINE_OPENERS:
            return ("markdown", cell)
        else:
            text += cell[i]
            i += 1
    return ("text", text)


def _gfm_row(line):
    """A GitHub table row as (kind, shown text) per cell, written from the GFM rules
    (_split_row, then _inline); the row must split alike under the parity rule of parsers that
    count backslashes."""
    cells = _split_row(line, "gfm")
    assert _split_row(line, "parity") == cells, line
    return [_inline(cell) for cell in cells]


def _registered_row(ident, definition):
    """The table row of a definition registered under ``ident`` for the test, read back by
    _gfm_row, with the number of lines of the whole table."""
    entry = ob.ObservableDef(
        ident, "1", "snapshot", ("physical_code",), {}, definition, ob.code_parameters
    )
    ob.register(entry)
    try:
        table = ob.catalog_markdown().splitlines()
        rows = [_gfm_row(line) for line in table[2:]]
        shown = " ".join(ident.splitlines()).strip(" \t")
        found = [cells for cells in rows if cells[0][1] == shown]
        assert len(found) == 1, table
        return found[0], len(table)
    finally:
        del ob._REGISTRY[(ident, "1")]


def test_the_catalog_table_keeps_backslashes_and_pipes():
    """A definition with a backslash before a pipe, a doubled backslash and a lone one reads back
    exactly (escaping only the pipe drops the backslash: "a\\|b" would show as "a|b"); so does an
    id with a pipe, in its code span, and an id with a backslash before a pipe, written as plain
    text (in a code span backslashes are literal, so doubling them would show two, and a single
    one before the escaped pipe is a run of two, which parsers that count the run do not read as
    an escape)."""
    hard = "P(A\\|m) and a\\|b, then c\\\\|d, a lone \\ and Q(A|m) (v1.4 §5.7)"
    cells, _ = _registered_row("t|able", hard)
    assert len(cells) == 6
    assert cells[0] == ("code", "t|able") and cells[5] == ("text", hard)
    cells, _ = _registered_row("t|ab\\|le", hard)
    assert cells[0] == ("text", "t|ab\\|le") and cells[5] == ("text", hard)


def test_the_catalog_table_shows_markdown_syntax_as_text():
    """Text that would be Markdown syntax in a cell (emphasis, code spans, links, images,
    angle brackets, HTML, entities, strikethrough, math) reads back as itself, as does an id with
    a backtick (which would end its code span); line breaks of every kind (\\n, \\r\\n, \\r) are
    joined by spaces, so the table keeps one line per row and each row its six cells. Addresses
    (``://``, ``www.``, e-mail) are not here: GitHub's extended autolinks read them from the raw
    text before any escape, so the registry refuses them (tested in
    test_register_get_and_versions)."""
    hard = (
        "*a* and **b**, _c_ and __d__, `e` and ``f``, [g](h), ![i](j), [k], <l>, <m.n>, "
        "&amp; and &#35;, ~o~ and ~~p~~, $q$ and $$r$$, s! and t] (v1.4 §5.7)\n"
        "second line\r\nthird line\rfourth line"
    )
    shown = " ".join(hard.splitlines())
    cells, lines = _registered_row("u`v", hard)
    assert lines == len(ob.catalog()) + 3  # the header, the rule, the rows and this one
    assert len(cells) == 6
    assert cells[0] == ("text", "u`v") and cells[5] == ("text", shown)
    # an identifier stays a code span; spaces and tabs at the ends go, as the table drops them
    # from a plain cell (a code span would keep a space at one end)
    cells, _ = _registered_row(" w_x", "\t *y*\t \n")
    assert cells[0] == ("code", "w_x") and cells[5] == ("text", "*y*")


def test_register_get_and_versions():
    entry = ob.get("d_cyc")
    assert ob.register(entry) is entry  # the same entry again: nothing changes
    changed = dataclasses.replace(entry, definition=entry.definition + " (changed)")
    with pytest.raises(ValueError, match="new version"):
        ob.register(changed)
    newer = dataclasses.replace(entry, version="2", definition="d_cyc, reimplemented (v1.4 §5.5)")
    ob.register(newer)
    try:
        assert ob.get("d_cyc") is newer and ob.get("d_cyc", "1") is entry
        assert {r["version"] for r in ob.catalog() if r["id"] == "d_cyc"} == {"1", "2"}
    finally:
        del ob._REGISTRY[("d_cyc", "2")]
    assert ob.get("d_cyc") is entry
    with pytest.raises(KeyError, match="no observable 'nope'"):
        ob.get("nope")
    with pytest.raises(KeyError, match="has no version '9'"):
        ob.get("d_cyc", "9")
    with pytest.raises(TypeError):
        ob.register("d_cyc")
    with pytest.raises(ValueError, match="level"):
        dataclasses.replace(entry, level="layer")
    with pytest.raises(ValueError, match="input"):
        dataclasses.replace(entry, inputs=("snapshot",))
    with pytest.raises(ValueError, match="input"):
        dataclasses.replace(entry, inputs=("trace",))
    # a bare address becomes a GitHub extended autolink from the raw text, which no escape can
    # stop, so the catalog table could not show it as written: such definitions are refused
    for text in ("see https://arxiv.org/abs/2007.03822", "see www.example.org/a_b", "x@y.org"):
        with pytest.raises(ValueError, match="bare web or e-mail address"):
            dataclasses.replace(entry, definition=text)
    assert not any(ob._BARE_ADDRESS.search(r["definition"]) for r in ob.catalog())
    with pytest.raises(ValueError, match="NUL"):
        dataclasses.replace(entry, id="d\x00cyc")
    with pytest.raises(ValueError, match="non-empty"):
        dataclasses.replace(entry, version="")


def test_parameters_are_resolved_and_keyed_canonically():
    assert ob.resolve_params("d_conn") == {"max_n": 20}
    assert ob.resolve_params("d_conn", {"max_n": 12}) == {"max_n": 12}
    with pytest.raises(ValueError, match=r"unknown parameter\(s\) \['n_max'\].*'max_n'"):
        ob.resolve_params("d_conn", {"n_max": 12})
    with pytest.raises(ValueError, match=r"needs the parameter\(s\) \['J', 'm'\]"):
        ob.resolve_params("mean_leakage")
    a = ob.params_key("mean_leakage", {"J": 1.0, "m": 3})
    b = ob.params_key("mean_leakage", {"m": 3, "samples": 100, "J": 1.0, "route": "sample"})
    assert a == b
    resolved = ob.resolve_params("mean_leakage", {"J": 1.0, "m": 3})
    assert a == json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    plateau = ob.params_key("peak_D")
    assert json.loads(plateau) == {
        "bootstrap": 200,
        "rule": "plateau",
        "sensitivity_z": [0.0, 1.0, 2.0],
        "z": 1.0,
    }
    assert ob.params_key("first_passage") == "{}"
    with pytest.raises(ValueError, match="couplings"):
        ob.params_key("D_J")


def test_numerically_equal_parameters_have_one_key():
    """The key of a stored value is one text per parameter set, whatever the numeric types."""
    one = ob.params_key("mean_leakage", {"m": 3, "J": 1.0})
    assert json.loads(one)["J"] == 1.0 and '"J":1.0' in one
    for j in (1, 1.0, np.float64(1), np.int64(1), np.float32(1)):
        assert ob.params_key("mean_leakage", {"m": 3, "J": j}) == one
    assert (
        ob.params_key("mean_leakage", {"m": np.int64(3), "J": 1, "samples": np.int32(100)}) == one
    )
    zero = ob.params_key("mean_leakage", {"m": 3, "J": 0.0})
    for j in (0, -0.0, np.float64(-0.0)):
        assert ob.params_key("mean_leakage", {"m": 3, "J": j}) == zero
    assert '"J":0.0' in zero
    couplings = ob.params_key("D_J", {"couplings": [0.0, 1.0, 2.0]})
    for c in ([0, 1, 2], (0, 1.0, 2), np.array([0.0, 1.0, 2.0]), np.arange(3)):
        assert ob.params_key("D_J", {"couplings": c}) == couplings
    assert ob.params_key("Phi", {"e": 0.1}) == ob.params_key("Phi", {"e": [0.1]})
    assert ob.params_key("D_J", {"couplings": [1], "sizes": [1, 2]}) == ob.params_key(
        "D_J", {"couplings": [1.0], "sizes": np.array([1, 2])}
    )
    # a range string of sizes is its expansion (Python slice semantics, no n): one key with the
    # list, as in the engine and the manifest (which stores the expansion)
    odd = ob.params_key("D_J", {"couplings": [0, 1.5], "sizes": [1, 3]})
    assert json.loads(odd)["sizes"] == [1, 3]
    for spelling in ("1:4:2", " 1 : 4 : 2 ", "1:5:2", (1, 3), np.array([1, 3])):
        assert ob.params_key("D_J", {"couplings": [0, 1.5], "sizes": spelling}) == odd
    for spelling, expansion in (("1:4", [1, 2, 3]), ("3:4", [3]), ("1:31:2", [*range(1, 31, 2)])):
        params = {"couplings": [1], "e": 0.1, "sizes": spelling}
        assert json.loads(ob.params_key("Phi_J", params))["sizes"] == expansion
    assert json.loads(ob.params_key("D_J", {"couplings": [0], "sizes": None}))["sizes"] is None
    for bad, message in (
        ("1:4:0", "step must be positive"),
        ("5:5", "at least one size"),
        ("a:b", "range like"),
        ("1-4", "range like"),
        ("", "range like"),
    ):
        with pytest.raises(ValueError, match=f"parameter 'sizes' of observable 'D_J': .*{message}"):
            ob.params_key("D_J", {"couplings": [0], "sizes": bad})
    # the manifest's entry and the file's spelling of it key alike (the store's lookup by either)
    raw = {"couplings": [0, 1.5], "sizes": "1:31:2", "samples": 8, "name": "ring"}
    parsed = CorrelatedErasure.from_dict(raw, "observables.correlated_erasure[0]")
    assert ob.params_key("D_J", parsed.to_dict()) == ob.params_key("D_J", raw)
    assert ob.params_key("correlated_failure", parsed.to_dict()) == ob.params_key(
        "correlated_failure", raw
    )
    # the design's manifest block analysis.peak (sensitivity_z as JSON ints) gives the default key
    manifest_peak = {"rule": "plateau", "z": 1, "sensitivity_z": [0, 1, 2], "bootstrap": 200}
    assert ob.params_key("peak_D", manifest_peak) == ob.params_key("peak_D")
    # resolved values are in canonical form, so the computation sees one value too
    resolved = ob.resolve_params("mean_leakage", {"m": np.int64(3), "J": 1})
    assert type(resolved["m"]) is int and type(resolved["J"]) is float
    # numeric parameters refuse strings, bools, fractions for ints and non-finite numbers
    for params, message in (
        ({"m": 3, "J": "1"}, "'J' of observable 'mean_leakage' must be a real number"),
        ({"m": 3, "J": True}, "real number"),
        ({"m": 3, "J": math.nan}, "finite"),
        ({"m": 3.0, "J": 1.0}, "'m' of observable 'mean_leakage' must be an integer"),
        ({"m": 3, "J": 1.0, "samples": 100.0}, "integer"),
        ({"m": 3, "J": 1.0, "tempering": 1}, "True or False"),
        ({"m": 3, "J": 1.0, "route": ""}, "non-empty string"),
        ({"m": 3, "J": 1.0, "ladder": [[0.0]]}, "1-D"),
    ):
        with pytest.raises(ValueError, match=re.escape(message)):
            ob.params_key("mean_leakage", params)
    with pytest.raises(ValueError, match=r"item 1\) must be a real number"):
        ob.params_key("D_J", {"couplings": [0.0, "1"]})
    key = TrajectoryKey("ws-test", 4, "0", "0.1", 0, 0)
    with pytest.raises(ValueError, match="must be a number"):  # the seed refuses a string J too
        ob.region_seed(key, 0, 1, "1", "mean_leakage", "1")
    # an integer beyond float64 is a ValueError naming the parameter, not an OverflowError
    for huge in (10**400, -(10**400)):
        with pytest.raises(ValueError, match="'J' of observable 'mean_leakage' must be finite"):
            ob.params_key("mean_leakage", {"m": 3, "J": huge})
        with pytest.raises(ValueError, match=r"'couplings' of observable 'D_J' \(item 1\)"):
            ob.params_key("D_J", {"couplings": [0, huge]})
        with pytest.raises(ValueError, match=r"\|J\| <= 1,000,000"):
            ob.region_seed(key, 0, 1, huge, "mean_leakage", "1")


def test_entries_declare_the_kind_of_every_parameter():
    entry = ob.get("d_conn")
    assert dict(entry.kinds) == {"max_n": "int"} and entry.row()["kinds"] == {"max_n": "int"}
    for e in (ob.get(r["id"], r["version"]) for r in ob.catalog()):
        assert set(e.kinds) == set(e.params)
        for name, kind in e.kinds.items():
            assert kind.removesuffix("?") in ob.KINDS and ob.PARAM_KINDS[name] == kind
    # defaults are stored in canonical form
    custom = ob.ObservableDef(
        "custom",
        "1",
        "trajectory",
        ("trace:D",),
        {"z": 1, "width": 2},
        "test (v1.4 §5.8)",
        ob.peak,
        kinds={"width": "real?"},
    )
    assert type(custom.params["z"]) is float and custom.params["width"] == 2.0
    assert custom.kinds == {"z": "real", "width": "real?"}
    for kwargs, message in (
        ({"params": {"width": 1}}, "has no kind"),
        ({"params": {"z": 1}, "kinds": {"width": "real"}}, "not parameters"),
        ({"params": {"width": 1}, "kinds": {"width": "complex"}}, "must be one of"),
        ({"params": {"z": "wide"}}, "default of parameter 'z'"),
    ):
        args = {
            "id": "custom",
            "version": "1",
            "level": "trajectory",
            "inputs": ("trace:D",),
            "definition": "test (v1.4 §5.8)",
            "function": ob.peak,
        }
        args.update(kwargs)
        with pytest.raises(ValueError, match=message):
            ob.ObservableDef(**args)


def test_peak_parameters_are_only_those_a_function_consumes():
    """peak_D's bootstrap is run by bootstrap_peak (orderings); D(J) has no orderings, so
    peak_D_J lists no bootstrap (its key would claim replicates that never ran). Its coupling J
    and the name of its correlated probe select the trace (two probes may share a coupling)."""
    assert set(ob.get("peak_D").params) == {"rule", "z", "sensitivity_z", "bootstrap"}
    assert set(ob.get("peak_D_J").params) == {"rule", "z", "sensitivity_z", "J", "name"}
    assert "bootstrap" not in json.loads(ob.params_key("peak_D_J", {"J": 1.0, "name": "ring"}))
    with pytest.raises(ValueError, match="name"):
        ob.params_key("peak_D_J", {"J": 1.0})
    assert "probe:erasure_first_failure" in ob.get("peak_D").inputs  # what the bootstrap reads


def test_compute_refuses_what_it_cannot_compute():
    code = StabilizerCode(["XXXX", "ZZZZ"])
    snap = ob.SnapshotInput(num_system=4, t=1, code=code)
    with pytest.raises(ValueError, match="first_passage"):
        ob.compute("first_passage", snap)
    with pytest.raises(ValueError, match="interaction graph"):
        ob.compute("d_conn", snap)
    with pytest.raises(ValueError, match="joint tableau"):
        ob.compute("half_cut_entropy", snap)
    with pytest.raises(ValueError, match="physical code"):
        ob.compute("k", ob.SnapshotInput(num_system=4, t=1))
    with pytest.raises(TypeError):
        ob.compute("k", {"code": code})
    key = TrajectoryKey("ws-test", 4, "0", "0.1", 0, 0)
    keyed = ob.SnapshotInput(4, 1, key=key, code=code, edges=graphs.cycle(4))
    with pytest.raises(ValueError, match="route"):
        ob.compute("mean_leakage", keyed, {"m": 2, "J": 0.0, "route": "guess"})
    unkeyed = ob.SnapshotInput(4, 1, code=code, edges=graphs.cycle(4))
    with pytest.raises(ValueError, match="give key="):
        ob.compute("mean_leakage", unkeyed, {"m": 2, "J": 0.0})
    with pytest.raises(ValueError, match="n = 5"):
        ob.SnapshotInput(4, 1, key=TrajectoryKey("ws-test", 5, "0", "0.1", 0, 0))
    with pytest.raises(ValueError, match="t must be"):
        ob.SnapshotInput(4, -1)


def test_compute_hands_the_entry_to_its_compute_and_keys_the_value():
    seen = []

    def spy(snapshot, params, entry):
        seen.append((entry.id, entry.version, dict(params)))
        return ob.ObservableValue(1.0, 0.0, "spy", {})

    entry = ob.ObservableDef(
        "spy", "7", "snapshot", ("physical_code",), {"m": 2}, "a test (v1.4 §5.2)", spy, spy
    )
    ob.register(entry)
    try:
        snap = ob.SnapshotInput(4, 1, code=StabilizerCode(["XXXX", "ZZZZ"]))
        value = ob.compute("spy", snap, {"m": np.int64(3)})
        assert seen == [("spy", "7", {"m": 3})]
        assert (value.observable_id, value.version, value.params_key) == ("spy", "7", '{"m":3}')
        broken = dataclasses.replace(entry, version="8", compute=lambda s, p, e: 1.0)
        ob.register(broken)
        with pytest.raises(TypeError, match="not an ObservableValue"):
            ob.compute("spy", snap, version="8")
    finally:
        ob._REGISTRY.pop(("spy", "7"), None)
        ob._REGISTRY.pop(("spy", "8"), None)


# ---------------------------------------------------------------------------------------------
# the k trace and first passage (v1.4 §5.2)
# ---------------------------------------------------------------------------------------------


def _reference_change_points(k):
    rows = [(0, k[0])] + [(t, k[t]) for t in range(1, len(k)) if k[t] < k[t - 1]]
    if rows[-1][0] != len(k) - 1:
        rows.append((len(k) - 1, k[-1]))
    return rows


def _random_trace(rng, n, horizon):
    """A nonincreasing trace from n with random multi-drops and plateaus."""
    k = [n]
    for _ in range(horizon):
        drop = 0 if rng.random() < 0.6 else int(rng.integers(1, 4))
        k.append(max(0, k[-1] - drop))
    return k


def test_change_points_round_trip_and_are_canonical(rng):
    cases = {
        (4,): [[0, 4]],
        (4, 4, 4): [[0, 4], [2, 4]],
        (4, 2, 2, 1): [[0, 4], [1, 2], [3, 1]],
        (3, 0): [[0, 3], [1, 0]],
        (5, 3, 0, 0, 0): [[0, 5], [1, 3], [2, 0], [4, 0]],
    }
    for k, points in cases.items():
        assert ob.k_change_points(list(k)).tolist() == points
        assert ob.k_from_change_points(points).tolist() == list(k)
    for _ in range(300):
        k = _random_trace(rng, int(rng.integers(0, 12)), int(rng.integers(0, 60)))
        points = ob.k_change_points(np.array(k, dtype=np.float64))
        assert points.dtype == np.int64
        assert [tuple(r) for r in points.tolist()] == _reference_change_points(k)
        assert ob.k_from_change_points(points).tolist() == k


def test_change_points_refuse_what_is_not_a_trace():
    with pytest.raises(ValueError, match=r"k\[2\] = 4 > k\[1\] = 3: layer 2 gained"):
        ob.k_change_points([3, 3, 4])
    for bad in ([3, math.nan], [3, 1.5], [3, -1], [], [[3, 2]]):
        with pytest.raises(ValueError):
            ob.k_change_points(bad)
    bad_points = {
        "layer 0": [[1, 3]],
        "increase strictly": [[0, 3], [2, 2], [2, 1]],
        "nonincreasing": [[0, 3], [2, 4]],
        "repeats": [[0, 3], [1, 3], [2, 2]],
        "shape": [0, 3],
    }
    for message, points in bad_points.items():
        with pytest.raises(ValueError, match=message):
            ob.k_from_change_points(points)


def test_first_passage_on_hand_made_traces():
    multi = ob.first_passage([6, 6, 6, 2, 2, 2, 2, 0])  # a drop of 4, then of 2
    assert multi.tau.tolist() == [7, 7, 3, 3, 3, 3, 0]
    assert multi.observed.all() and multi.horizon == 7 == multi.censor_depth
    censored = ob.first_passage([4, 4, 3, 3, 3])
    assert censored.tau.tolist() == [-1, -1, -1, 2, 0]
    assert censored.observed.tolist() == [False, False, False, True, True]
    assert censored.survival_data(1) == (4, False) and censored.survival_data(3) == (2, True)
    assert censored.to_dict()["tau"] == [None, None, None, 2, 0]
    with pytest.raises(ValueError, match="ell"):
        censored.survival_data(5)
    # stop_when_purified fills k = 0 after the stop: the passage times do not change
    assert (
        ob.first_passage([3, 2, 0, 0, 0, 0]).tau.tolist()
        == ob.first_passage([3, 2, 0]).tau.tolist()
    )
    assert ob.first_passage([2, 1], num_system=4).tau.tolist() == [-1, 1, 0, 0, 0]
    with pytest.raises(ValueError, match="exceeds num_system"):
        ob.first_passage([5, 1], num_system=4)
    lone = ob.first_passage([3])
    assert lone.tau.tolist() == [-1, -1, -1, 0] and lone.horizon == 0
    chk = ob.first_passage([5, 5, 2, 2, 2, 1], checkpoints=[0, 4, 5])
    assert chk.tau.tolist() == [-1, 5, 2, 2, 2, 0]
    assert chk.tau_chk.tolist() == [-1, 5, 4, 4, 4, 0]
    assert chk.observed_chk.tolist() == [False, True, True, True, True, True]
    assert chk.to_dict()["tau_chk"] == [None, 5, 4, 4, 4, 0]
    sparse = ob.first_passage(
        [5, 5, 2, 2, 2, 1], checkpoints=[0, 1]
    )  # no checkpoint after the drops
    assert sparse.tau_chk.tolist() == [-1, -1, -1, -1, -1, 0]
    from_points = ob.first_passage(ob.k_change_points([5, 5, 2, 2, 2, 1]), checkpoints=[0, 4, 5])
    assert from_points.tau.tolist() == chk.tau.tolist() and from_points.horizon == 5
    with pytest.raises(ValueError, match="beyond the horizon"):
        ob.first_passage([3, 2], checkpoints=[0, 5])
    with pytest.raises(ValueError, match="increase strictly"):
        ob.first_passage([3, 2, 1], checkpoints=[2, 1])


def test_first_passage_matches_its_definition(rng):
    for _ in range(300):
        n = int(rng.integers(0, 10))
        k = _random_trace(rng, n, int(rng.integers(0, 40)))
        horizon = len(k) - 1
        cps = sorted(
            {0, horizon, *rng.integers(0, horizon + 1, size=int(rng.integers(0, 6))).tolist()}
        )
        fp = ob.first_passage(k, checkpoints=cps)
        for ell in range(n + 1):
            hits = [t for t in range(horizon + 1) if k[t] <= ell]
            hits_chk = [t for t in cps if k[t] <= ell]
            assert fp.observed[ell] == bool(hits)
            assert fp.tau[ell] == (hits[0] if hits else -1)
            assert fp.tau_chk[ell] == (hits_chk[0] if hits_chk else -1)
            if fp.observed[ell] and fp.observed_chk[ell]:
                assert fp.tau[ell] <= fp.tau_chk[ell]  # Proposition 3
        for t in range(horizon + 1):  # Proposition 2: the passage times determine the trace
            assert k[t] == min(ell for ell in range(n + 1) if fp.observed[ell] and fp.tau[ell] <= t)


@pytest.mark.parametrize(
    "n, beta, p",
    [(5, 0.0, 0.2), (7, 0.1, 0.15), (8, 1.0, 0.3), (11, 0.1, 0.1), (16, 0.3, 0.12)],
)
def test_first_passage_on_engine_traces(n, beta, p):
    layers = 4 * n
    stopped = _run(n, beta, p, layers, [], stop=True, circuits=3)
    running = _run(n, beta, p, layers, [], stop=False, circuits=3)
    k_stop = stopped.values["code_dimension"]
    # the zero fill after a stop is the exact value: the traces are equal (Proposition 4)
    np.testing.assert_array_equal(k_stop, running.values["code_dimension"])
    cps = sorted({0, 1, layers, *(2**j for j in range(int(math.log2(layers)) + 1))})
    for row in range(k_stop.shape[0]):
        fp = ob.first_passage(k_stop[row], checkpoints=cps)
        assert fp.num_system == n and fp.tau[n] == 0
        expected = stopped.purification_layer[row]
        assert fp.tau[0] == expected  # tau_0 is the engine's purification layer, or -1 (censored)
        assert fp.observed[0] == (expected >= 0)
        both = fp.observed & fp.observed_chk
        assert np.all(fp.tau[both] <= fp.tau_chk[both])
        assert np.array_equal(fp.observed_chk, fp.observed)  # T_max is a checkpoint
        points = ob.k_change_points(k_stop[row])
        assert np.array_equal(ob.k_from_change_points(points), k_stop[row].astype(np.int64))


def test_code_parameters():
    k, rate, rank = ob.code_parameters([4, 2, 0], 4)
    assert (
        k.tolist() == [4, 2, 0] and rate.tolist() == [1.0, 0.5, 0.0] and rank.tolist() == [0, 2, 4]
    )
    with pytest.raises(ValueError, match="exceeds n = 4"):
        ob.code_parameters([5], 4)


# ---------------------------------------------------------------------------------------------
# d_cyc (v1.4 §5.5)
# ---------------------------------------------------------------------------------------------


def test_cyclic_distance_forms_and_refusals():
    d = ob.cyclic_distance([3, math.nan, 1], [2, 0, 1], num_system=6)
    assert d.zero_extended.tolist() == [3, 0, 1] and d.has_code.tolist() == [True, False, True]
    assert d.raw[0] == 3 and math.isnan(d.raw[1])
    for raw, k, message in (
        ([math.nan], [1], "whole number"),
        ([0.0], [1], "whole number"),
        ([1.5], [1], "whole number"),
        ([7.0], [1], "whole number"),
        ([2.0], [0], "records NaN"),
        ([2.0, 3.0], [1], "same shape"),
    ):
        with pytest.raises(ValueError, match=message):
            ob.cyclic_distance(raw, k, num_system=6)


@pytest.mark.parametrize(
    "n, beta, p, layers",
    [
        (3, 0.0, 0.2, 9),
        (4, 0.0, 0.15, 12),
        (5, 0.3, 0.2, 15),
        (8, 0.1, 0.1, 24),
        (11, 1.0, 0.1, 22),
        (16, 0.1, 0.08, 24),
    ],
)
def test_cyclic_distance_of_engine_codes_three_ways(n, beta, p, layers):
    cps = sorted({0, 1, 2, layers // 3, layers // 2, layers})
    result = _run(
        n,
        beta,
        p,
        layers,
        [mipt.contiguous_distance(checkpoints=cps), mipt.tableau_snapshot(checkpoints=cps)],
    )
    raw_all = result.values["contiguous_distance"]
    k_all = result.values["code_dimension"][:, cps]
    forms = ob.cyclic_distance(raw_all, k_all, num_system=n)
    assert np.array_equal(forms.has_code, k_all > 0)
    for row in range(raw_all.shape[0]):
        for e in range(len(cps)):
            raw = raw_all[row, e]
            from_code = ob.cyclic_distance_from_code(result.code(row, e))
            assert (math.isnan(raw) and math.isnan(from_code)) or raw == from_code
            if not math.isnan(raw):
                assert raw <= (n + 1) // 2  # Proposition 10: d_cyc <= (n + 1) / 2
            if n <= 5:
                joint = _joint_oracle(result, row, e)
                table = oracle.info_table(joint.info, n)
                reference = oracle.cyclic_distance(table, n, joint.k)
                assert (math.isnan(raw) and math.isnan(reference)) or raw == reference


# ---------------------------------------------------------------------------------------------
# reference information, correctability and the identities of v1.4 §5.8
# ---------------------------------------------------------------------------------------------


def _check_every_region(n, joint, code, tableau):
    """I(A:R) three ways, the complement identity, correctability, monotonicity, for every A."""
    k = joint.k
    assert code.num_logical == k
    full = (1 << n) - 1
    reference = list(range(n, 2 * n))
    for mask in range(1 << n):
        region = _qubits(mask, n)
        info = joint.info(mask)
        assert code.lost_information(region) == info
        if tableau is not None:
            assert tableau.mutual_information(region, reference) == info
        assert info + joint.info(full ^ mask) == 2 * k  # v1.4 §5.8
        assert code.is_correctable(region) == (info == 0)  # f(A) = 0 iff correctable
        for q in range(n):  # f is monotone under inclusion
            if not (mask >> q) & 1:
                assert (info > 0) <= (joint.info(mask | (1 << q)) > 0)


def test_reference_information_identities_on_random_codes(rng):
    checked = 0
    for n in range(1, 6):
        for _ in range(4):
            measured = rng.choice(n, size=int(rng.integers(0, n + 1)), replace=False)
            rows = oracle.random_joint_state(n, rng, measured)
            joint = oracle.Joint(rows, n)
            tableau = PhaseFreeTableau(oracle.ints_to_dense(rows, 2 * n))
            code = StabilizerCode(
                oracle.ints_to_dense(joint.physical_generators(), n), num_qubits=n
            )
            _check_every_region(n, joint, code, tableau)
            assert tableau.physical_code(n).num_logical == joint.k
            # the oracle's two routes agree: entropies of the joint state and dim N_A - dim S_A
            by_code = oracle.Code(joint.physical_generators(), n)
            assert all(by_code.lost(m) == joint.info(m) for m in range(1 << n))
            checked += 1
    assert checked == 20


@pytest.mark.parametrize(
    "n, beta, p", [(3, 0.0, 0.2), (4, 0.0, 0.1), (5, 0.5, 0.15), (6, 0.2, 0.1), (7, 0.1, 0.08)]
)
def test_reference_information_identities_on_engine_codes(n, beta, p):
    cps = sorted({0, 1, 3, n, 2 * n})
    result = _run(n, beta, p, 2 * n, [mipt.tableau_snapshot(checkpoints=cps)], graphs_count=1)
    for row in range(result.num_trajectories):
        for e in range(len(cps)):
            joint = _joint_oracle(result, row, e)
            _check_every_region(n, joint, result.code(row, e), result.tableau(row, e))


def test_complementary_failure_and_robustness_bounds_on_engine_codes():
    """P(m) + P(n-m) >= 1 when k > 0 and D(J) <= (n + 1) / 2, on the exact profiles of real codes."""
    couplings = [-0.5, 0.0, 0.7, 2.0]
    seen_code = 0
    for n, beta in ((6, 0.0), (7, 0.3), (8, 0.1)):
        cps = [1, n, 2 * n]
        result = _run(n, beta, 0.1, 2 * n, [mipt.tableau_snapshot(checkpoints=cps)])
        for row in range(result.num_trajectories):
            edges = result.spec.graphs[result.graph_index[row]]
            for e in range(len(cps)):
                code = result.code(row, e)
                k = code.num_logical
                exact = code.boundary_failure_counts(edges, n).exact(couplings)  # (c, n): m = 1..n
                curve = np.concatenate([np.zeros((len(couplings), 1)), exact], axis=1)
                joint = _joint_oracle(result, row, e)
                table = oracle.info_table(joint.info, n)
                for i, coupling in enumerate(couplings):
                    reference = oracle.correlated_failure(table, edges.tolist(), n, coupling)
                    np.testing.assert_allclose(curve[i], reference, atol=1e-12)
                uniform = [float(x) for x in oracle.uniform_failure(table, n)]
                np.testing.assert_allclose(curve[couplings.index(0.0)], uniform, atol=1e-15)
                if k == 0:
                    assert not curve.any()
                    continue
                seen_code += 1
                for m in range(n + 1):
                    assert np.all(curve[:, m] + curve[:, n - m] >= 1 - 1e-12)
                d_j = (1 - curve[:, :n]).sum(axis=1)
                assert np.all(d_j <= (n + 1) / 2 + 1e-12)
    assert seen_code >= 20


def test_leakage_identities_and_exact_values_on_engine_codes():
    couplings = [-0.5, 0.0, 0.7, 2.0]
    for n, beta in ((6, 0.0), (7, 0.3), (8, 0.1)):
        cps = [1, n, 2 * n]
        result = _run(n, beta, 0.1, 2 * n, [mipt.tableau_snapshot(checkpoints=cps)], graphs_count=1)
        for row in range(result.num_trajectories):
            edges = result.spec.graphs[result.graph_index[row]]
            for e in range(len(cps)):
                code = result.code(row, e)
                k = code.num_logical
                table = oracle.info_table(_joint_oracle(result, row, e).info, n)
                exact = np.array(
                    [ob.mean_leakage_exact(code, edges, m, couplings) for m in range(n + 1)]
                )
                for m in range(n + 1):
                    for i, coupling in enumerate(couplings):
                        reference = oracle.mean_leakage(table, edges.tolist(), n, m, coupling)
                        assert abs(exact[m, i] - reference) < 1e-9
                    np.testing.assert_allclose(exact[m] + exact[n - m], 2 * k, atol=1e-9)
                np.testing.assert_allclose(exact[0], 0.0, atol=0)
                np.testing.assert_allclose(exact[n], 2 * k, atol=1e-12)
                if n % 2 == 0:
                    np.testing.assert_allclose(exact[n // 2], k, atol=1e-9)
    with pytest.raises(ValueError, match="above the budget"):
        ob.mean_leakage_exact(StabilizerCode(["XXXX", "ZZZZ"]), graphs.cycle(4), 2, [0.0], budget=5)


# ---------------------------------------------------------------------------------------------
# uniform erasures: P_C(m) and D(0) (v1.4 §5.7)
# ---------------------------------------------------------------------------------------------


def test_uniform_failure_profile():
    prof = ob.uniform_failure([[2, 2, 3, 1]], 4)
    assert prof.curve.tolist() == [[0.0, 0.25, 0.75, 1.0, 1.0]]
    assert prof.exact.tolist() == [True, False, False, False, True]
    expected_se = [0.0, math.sqrt(0.25 * 0.75 / 4), math.sqrt(0.75 * 0.25 / 4), 0.0, 0.0]
    np.testing.assert_allclose(prof.se[0], expected_se, rtol=1e-15)
    hybrid = ob.uniform_failure([2, 2, 3, 1], 4, exact_counts=[1])  # one failing single qubit
    assert hybrid.curve.tolist() == [0.0, 0.25, 0.75, 1.0, 1.0]
    assert hybrid.exact.tolist() == [True, True, False, False, True] and hybrid.max_exact == 1
    assert hybrid.se[1] == 0.0
    none = ob.uniform_failure([[5, 5, 5]], 4)  # k = 0: no ordering fails
    assert not none.curve.any() and not none.se.any()
    with pytest.raises(ValueError, match="mixes n \\+ 1"):
        ob.uniform_failure([2, 5, 3], 4)
    with pytest.raises(ValueError, match=r"\[1, 5\]"):
        ob.uniform_failure([0, 2], 4)
    with pytest.raises(ValueError, match="at least one ordering"):
        ob.uniform_failure(np.zeros((2, 0)), 4)


def _reference_robustness(first, n, counts=None):
    """D(0) by a loop over m from the definition of the hybrid curve, and the Y-based error."""
    samples = len(first)
    if min(first) == n + 1:
        return 0.0, 0.0
    w = 0 if counts is None else len(counts)
    total = 0.0
    for m in range(n):
        if m == 0:
            p = 0.0
        elif m <= w:
            p = counts[m - 1] / math.comb(n, m)
        else:
            p = sum(1 for d in first if d <= m) / samples
        total += 1 - p
    y = [max(d, w + 1) - (w + 1) for d in first]
    mean = sum(y) / samples
    se = (
        math.sqrt(sum((v - mean) ** 2 for v in y) / (samples - 1) / samples)
        if samples > 1
        else math.nan
    )
    return total, se


def test_integrated_robustness_formulas(rng):
    n = 7
    for _ in range(50):
        first = rng.integers(1, n + 1, size=int(rng.integers(2, 30))).tolist()
        r = ob.integrated_robustness(first, n)
        value, se = _reference_robustness(first, n)
        assert abs(float(r.value) - value) < 1e-12 and abs(float(r.se) - se) < 1e-12
        assert abs(value - sum(first) / len(first)) < 1e-12  # the plain estimator is the mean of D
        w = int(rng.integers(1, n + 1))
        # nonzero counts never contradict the samples (a zero count at m < min D would be allowed too)
        counts = [int(rng.integers(1, math.comb(n, m) + 1)) for m in range(1, w + 1)]
        r = ob.integrated_robustness(first, n, exact_counts=counts)
        value, se = _reference_robustness(first, n, counts)
        assert abs(float(r.value) - value) < 1e-12 and abs(float(r.se) - se) < 1e-12
        if w >= n - 1:
            assert float(r.se) == 0.0  # every term exact
    with pytest.raises(ValueError, match="contradicts"):
        ob.integrated_robustness(
            [3, 5, 4], n, exact_counts=[0, 0, 0]
        )  # no 3-region fails, yet D = 3
    with pytest.raises(ValueError, match="contradicts"):
        ob.integrated_robustness([8, 8], n, exact_counts=[0, 1])  # k = 0 with a failing pair
    with pytest.raises(ValueError, match="shape"):
        ob.integrated_robustness([3, 5], n, exact_counts=[[0], [1]])
    assert float(
        ob.integrated_robustness([3, 5, 4], n, exact_counts=[0, 0]).value
    ) == pytest.approx(1 + 1 + 1 + (0 + 2 + 1) / 3)
    assert float(ob.integrated_robustness([3, 5], n, k=2).value) == 4.0
    with pytest.raises(ValueError, match="contradicts first_failure"):
        ob.integrated_robustness([3, 5], n, k=0)
    with pytest.raises(ValueError, match="contradicts first_failure"):
        ob.uniform_failure([[8, 8]], n, k=[1])
    with pytest.raises(ValueError, match="one per code"):
        ob.uniform_failure([[8, 8]], n, k=[0, 0])
    none = ob.integrated_robustness([[8, 8, 8]], n)
    assert none.value.tolist() == [0.0] and none.se.tolist() == [0.0] and not none.has_code[0]
    lone = ob.integrated_robustness([3], n)
    assert float(lone.value) == 3.0 and math.isnan(float(lone.se))
    stacked = rng.integers(1, n + 1, size=(2, 3, 9))
    batch = ob.integrated_robustness(stacked, n)
    for i in range(2):
        for j in range(3):
            single = ob.integrated_robustness(stacked[i, j], n)
            assert batch.value[i, j] == single.value and batch.se[i, j] == single.se


def _binomial_weights(n, e):
    return [math.comb(n, m) * e**m * (1 - e) ** (n - m) for m in range(n + 1)]


def _binomial_tail(n, e, d):
    return sum(_binomial_weights(n, e)[d:]) if d <= n else 0.0


def test_uniform_channel_failure_formulas(rng):
    n, es = 7, [0.0, 0.05, 0.3, 1.0]
    for _ in range(30):
        first = rng.integers(1, n + 1, size=int(rng.integers(2, 25))).tolist()
        w = int(rng.integers(0, n + 1))
        counts = [int(rng.integers(1, math.comb(n, m) + 1)) for m in range(1, w + 1)] if w else None
        phi = ob.uniform_channel_failure(first, n, es, exact_counts=counts)
        for i, e in enumerate(es):
            weights = _binomial_weights(n, e)
            curve = [0.0] + [
                counts[m - 1] / math.comb(n, m)
                if m <= w
                else sum(1 for d in first if d <= m) / len(first)
                for m in range(1, n + 1)
            ]
            assert (
                abs(phi.value[i] - sum(b * q for b, q in zip(weights, curve, strict=True))) < 1e-12
            )
            z = [_binomial_tail(n, e, max(d, w + 1)) for d in first]
            mean = sum(z) / len(z)
            se = math.sqrt(sum((v - mean) ** 2 for v in z) / (len(z) - 1) / len(z))
            assert abs(phi.se[i] - se) < 1e-12
        assert not phi.interpolated_mass.any() and phi.e.tolist() == es
        assert phi.se.shape == phi.value.shape == (len(es),)
    gone = ob.uniform_channel_failure([[8, 8, 8]], n, 0.5)
    assert gone.value.tolist() == [[0.0]] and gone.se.tolist() == [[0.0]]
    lone = ob.uniform_channel_failure([3], n, [0.5])
    assert math.isnan(float(lone.se[0]))
    # one ordering and several rates: an error per rate (NaN: no spread from one ordering)
    lone = ob.uniform_channel_failure([3], n, [0.1, 0.5])
    assert lone.value.shape == lone.se.shape == (2,) and np.isnan(lone.se).all()
    stacked = ob.uniform_channel_failure([[3], [8], [5]], n, [0.1, 0.3, 0.5])  # k = 0 in row 1
    assert stacked.value.shape == stacked.se.shape == (3, 3)
    assert np.isnan(stacked.se[[0, 2]]).all() and stacked.se[1].tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="probability"):
        ob.uniform_channel_failure([3, 4], n, 1.5)


def test_channel_failure_never_shares_or_freezes_the_callers_rates():
    """The result is read-only, but its rates are a copy: the caller's array stays writable and
    a later edit of it (or of the array it is a view of) does not change the stored rates."""
    first = [[2, 3, 4, 3], [5, 5, 3, 2]]
    e = np.array([0.1, 0.3])
    phi = ob.uniform_channel_failure(first, 6, e)
    assert e.flags.writeable and not np.shares_memory(phi.e, e)
    e[0] = 0.2
    assert phi.e.tolist() == [0.1, 0.3] and not phi.e.flags.writeable
    profile = ob.correlated_failure([[0.5, 1.0]], [0.0], [2, 4], 6, 1, samples=4)
    base = np.array([0.0, 0.1, 0.3, 0.9])
    channel = ob.correlated_channel_failure(profile, base[1:3])
    assert base.flags.writeable and not np.shares_memory(channel.e, base)
    base[1:3] = 0.5
    assert channel.e.tolist() == [0.1, 0.3]


def test_uniform_channel_failure_memory_does_not_grow_with_the_rate_count():
    """The per-ordering terms Z_j are formed one rate at a time. With 40 rates the former dense
    (codes, S, q) array and its variance temporary peaked at 82 times the input (measured); the
    peak is now a few input copies (measured: 5.1 times, as uniform_failure's 5.0), and the bound
    is 12 times. Allocation sizes are deterministic (no timing involved)."""
    n, rates = 16, np.linspace(0.01, 0.6, 40)
    rng = np.random.default_rng(3)
    first = rng.integers(1, n + 1, size=(64, 200))
    ob.uniform_channel_failure(first[:2], n, rates)  # warm up caches and lazy imports
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        phi = ob.uniform_channel_failure(first, n, rates)
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()
    assert phi.se.shape == (64, 40)
    assert peak <= 12 * first.nbytes, peak / first.nbytes


def test_uniform_channel_failure_matches_the_exact_value(rng):
    """|Phi_hat(e) - Phi(e)| <= 4.5 sd, sd = sqrt(Var(Z) / S) exactly from the oracle's distribution
    of M (Z = T_e(max(M, w + 1))); S = 400 orderings, 8 codes x 3 rates x 2 estimators = 48
    comparisons at 6.8e-6 each (normal approximation): below 3.3e-4 in all."""
    n, samples, es = 6, 400, [0.1, 0.25, 0.5]
    compared = 0
    while compared < 8:
        measured = rng.choice(n, size=int(rng.integers(0, 3)), replace=False)
        joint = oracle.Joint(oracle.random_joint_state(n, rng, measured), n)
        if joint.k == 0:
            continue
        table = oracle.info_table(joint.info, n)
        exact_curve = [float(x) for x in oracle.uniform_failure(table, n)]
        dist = oracle.first_failure_distribution(table, n)
        code = StabilizerCode(oracle.ints_to_dense(joint.physical_generators(), n), num_qubits=n)
        first = code.first_failure_sizes(samples, seed=[11, compared])
        for w in (0, 2):
            counts = code.uncorrectable_counts(w) if w else None
            phi = ob.uniform_channel_failure(first, n, es, exact_counts=counts, k=joint.k)
            for i, e in enumerate(es):
                exact = sum(
                    b * q for b, q in zip(_binomial_weights(n, e), exact_curve, strict=True)
                )
                _, var_z = _exact_moments(
                    dist, lambda m, e=e, w=w: _binomial_tail(n, e, max(m, w + 1))
                )
                assert (
                    abs(phi.value[i] - exact) <= 4.5 * math.sqrt(max(var_z, 0.0) / samples) + 1e-12
                )
        compared += 1


def _exact_moments(distribution, transform):
    mean = sum(p * transform(m) for m, p in enumerate(distribution))
    second = sum(p * transform(m) ** 2 for m, p in enumerate(distribution))
    return mean, second - mean**2


def test_D0_is_the_expected_first_failure_size_exactly(rng):
    """D(0) = sum_{m<n} (1 - P_C(m)) = E_pi[M_pi], and Pr(M_pi <= m) = P_C(m), over all n! orderings."""
    cases = 0
    for n in range(1, 7):
        for _ in range(3):
            measured = rng.choice(n, size=int(rng.integers(0, n + 1)), replace=False)
            joint = oracle.Joint(oracle.random_joint_state(n, rng, measured), n)
            table = oracle.info_table(joint.info, n)
            curve = oracle.uniform_failure(table, n)
            dist = oracle.first_failure_distribution(table, n)
            for m in range(n + 1):
                assert sum(dist[: m + 1]) == curve[m]  # the prefix of a random ordering is uniform
            if joint.k == 0:
                assert dist[n + 1] == 1 and oracle.robustness(curve, n, 0) == 0
                continue
            expected = sum(m * p for m, p in enumerate(dist))
            assert dist[n + 1] == 0 and dist[0] == 0
            assert oracle.robustness(curve, n, joint.k) == expected
            assert expected <= Fraction(n + 1, 2)  # v1.4 §5.8
            cases += 1
    assert cases >= 10


def test_D0_estimates_match_the_exact_value(rng):
    """|D_hat - D| <= 4.5 sd for plain and hybrid estimates of random codes.

    sd is the exact standard deviation sqrt(Var(M) / S) (or sqrt(Var(Y) / S)) from the oracle's
    distribution of M over all orderings; S = 400 independent orderings, so D_hat is close to
    normal: each comparison has a false-alarm probability of about 7e-6, below 2e-4 for the 24.
    """
    samples, w = 400, 2
    compared = 0
    while compared < 12:
        n = 6
        measured = rng.choice(n, size=int(rng.integers(0, 3)), replace=False)
        rows = oracle.random_joint_state(n, rng, measured)
        joint = oracle.Joint(rows, n)
        if joint.k == 0:
            continue
        table = oracle.info_table(joint.info, n)
        dist = oracle.first_failure_distribution(table, n)
        exact = float(sum(m * p for m, p in enumerate(dist)))
        code = StabilizerCode(oracle.ints_to_dense(joint.physical_generators(), n), num_qubits=n)
        first = code.first_failure_sizes(samples, seed=[7, compared])
        plain = ob.integrated_robustness(first, n)
        _, variance = _exact_moments(dist, float)
        assert abs(float(plain.value) - exact) <= 4.5 * math.sqrt(variance / samples) + 1e-12
        counts = code.uncorrectable_counts(w)
        hybrid = ob.integrated_robustness(first, n, exact_counts=counts)
        _, var_y = _exact_moments(dist, lambda m: max(m, w + 1) - (w + 1))
        assert abs(float(hybrid.value) - exact) <= 4.5 * math.sqrt(var_y / samples) + 1e-12
        assert var_y <= variance + 1e-12  # the exact terms remove noise
        compared += 1


def test_D0_from_the_engine_probes_matches_the_exact_value():
    """Engine first-failure samples and exact counts against the oracle's exact D(0) per snapshot.

    Same tolerance as above (4.5 exact standard deviations, about 7e-6 per comparison); the run
    has at most 4 x 3 = 12 snapshots with a code, so the total false-alarm rate is below 1e-4.
    """
    n, samples, w = 6, 300, 2
    cps = [2, 6, 12]
    result = _run(
        n,
        0.2,
        0.08,
        12,
        [
            mipt.erasure_first_failure(samples, checkpoints=cps),
            mipt.uncorrectable_region_counts(w, checkpoints=cps),
            mipt.tableau_snapshot(checkpoints=cps),
        ],
    )
    first = result.values["erasure_first_failure"]
    counts = result.values[f"uncorrectable_region_counts[{w}]"]
    estimate = ob.integrated_robustness(first, n, exact_counts=counts)
    k = result.values["code_dimension"][:, cps]
    assert np.array_equal(estimate.has_code, k > 0)
    tested = 0
    for row in range(result.num_trajectories):
        for e in range(len(cps)):
            table = oracle.info_table(_joint_oracle(result, row, e).info, n)
            if k[row, e] == 0:
                assert estimate.value[row, e] == 0.0
                continue
            dist = oracle.first_failure_distribution(table, n)
            exact = float(sum(m * p for m, p in enumerate(dist)))
            _, var_y = _exact_moments(dist, lambda m: max(m, w + 1) - (w + 1))
            assert abs(estimate.value[row, e] - exact) <= 4.5 * math.sqrt(var_y / samples) + 1e-12
            exact_counts = [
                sum(1 for mask in oracle.masks_of_size(n, m) if table[mask] > 0)
                for m in range(1, w + 1)
            ]
            assert counts[row, e].tolist() == exact_counts
            tested += 1
    assert tested >= 6


# ---------------------------------------------------------------------------------------------
# geometry-correlated erasures: P_{C,G,J}(m) and D(J) (v1.4 §5.7)
# ---------------------------------------------------------------------------------------------


def test_correlated_profile_bookkeeping():
    n, samples = 5, 4
    values = np.array([[0.25, 0.5, 0.75, 1.0], [0.0, 0.25, 0.5, 1.0]])
    prof = ob.correlated_failure(values, [0.0, 1.0], [1, 2, 3, 4], n, 2, samples=samples)
    assert prof.complete and prof.interpolated_sizes.size == 0
    assert prof.curve.tolist() == [
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.0],
        [0.0, 0.0, 0.25, 0.5, 1.0, 1.0],
    ]
    np.testing.assert_allclose(prof.robustness, [1 + 0.75 + 0.5 + 0.25 + 0.0, 1 + 1 + 0.75 + 0.5])
    se = np.sqrt(values * (1 - values) / samples)
    np.testing.assert_allclose(prof.se, se)
    np.testing.assert_allclose(prof.robustness_se, np.sqrt((se**2).sum(axis=1)))
    # sizes 1 and 3 only: P(2) = (P1 + P3) / 2 and P(4) = (P3 + 1) / 2, so the weights are 1.5 and 2
    part = ob.correlated_failure(values[:, [0, 2]], [0.0, 1.0], [1, 3], n, 2, samples=samples)
    assert not part.complete and part.interpolated_sizes.tolist() == [2, 4]
    p1, p3 = values[:, 0], values[:, 2]
    np.testing.assert_allclose(part.robustness, n - (1.5 * p1 + 2 * p3 + 0.5))
    np.testing.assert_allclose(
        part.robustness_se, np.sqrt(2.25 * se[:, 0] ** 2 + 4.0 * se[:, 2] ** 2)
    )
    # Phi_J: a_1 = b_1 + b_2 / 2, a_3 = b_2 / 2 + b_3 + b_4 / 2; the interpolated mass is b_2 + b_4
    channel = ob.correlated_channel_failure(part, [0.0, 0.2, 0.6])
    for q, e in enumerate([0.0, 0.2, 0.6]):
        b = _binomial_weights(n, e)
        expected = b[1] * p1 + b[2] * (p1 + p3) / 2 + b[3] * p3 + b[4] * (p3 + 1) / 2 + b[5]
        np.testing.assert_allclose(channel.value[:, q], expected, rtol=1e-12, atol=1e-15)
        a1, a3 = b[1] + b[2] / 2, b[2] / 2 + b[3] + b[4] / 2
        np.testing.assert_allclose(
            channel.se[:, q],
            np.sqrt(a1**2 * se[:, 0] ** 2 + a3**2 * se[:, 2] ** 2),
            rtol=1e-12,
            atol=1e-15,
        )
        assert channel.interpolated_mass[q] == pytest.approx(b[2] + b[4], rel=1e-12, abs=1e-15)
    with pytest.raises(TypeError):
        ob.correlated_channel_failure(values, 0.1)
    # leading axes, k = 0 rows, the sampler carried read-only
    stack = np.stack([values, np.zeros_like(values)])
    batch = ob.correlated_failure(
        stack, [0.0, 1.0], [1, 2, 3, 4], n, [2, 0], samples=samples, sampler={"method": "exact"}
    )
    assert batch.robustness[1].tolist() == [0.0, 0.0] and batch.robustness_se[1].tolist() == [
        0.0,
        0.0,
    ]
    np.testing.assert_allclose(batch.robustness[0], prof.robustness)
    gone = ob.correlated_channel_failure(batch, [0.3])
    assert gone.value[1].tolist() == [[0.0], [0.0]] and gone.se[1].tolist() == [[0.0], [0.0]]
    assert batch.sampler["method"] == "exact"
    with pytest.raises(TypeError):
        batch.sampler["method"] = "chain"
    # se_exact: only an exact sampler makes the binomial errors the true ones
    assert batch.se_exact and not prof.se_exact  # prof was given no sampler: not known exact
    for method in ("chain", "mixed"):
        chain = ob.correlated_failure(
            values, [0.0, 1.0], [1, 2, 3, 4], n, 2, samples=samples, sampler={"method": method}
        )
        assert not chain.se_exact
    for kwargs, message in (
        ({"values": values[:, :3]}, "shape"),
        ({"values": values * 0.9}, "multiples"),
        ({"values": values + 0.5}, r"\[0, 1\]"),
        ({"k": 0}, "nonzero"),
        ({"k": [2, 2]}, "leading shape"),
        ({"couplings": [0.0, 0.0]}, "distinct"),
        ({"sizes": [1, 1, 2, 3]}, "distinct sizes"),
    ):
        args = {
            "values": values,
            "couplings": [0.0, 1.0],
            "sizes": [1, 2, 3, 4],
            "num_system": n,
            "k": 2,
        }
        args.update(kwargs)
        with pytest.raises(ValueError, match=message):
            ob.correlated_failure(samples=samples, **args)


@pytest.mark.parametrize("n", [6, 7, 8])
def test_correlated_probe_matches_the_exact_law(n):
    """Exact (cycle) samples: every P_hat_J(m) and D_hat(J) against the oracle's exact values.

    Each P_hat is a Bin(S, P) fraction (independent exact samples): its exact two-sided binomial
    p-value must exceed 1e-4 / (number of comparisons) (Bonferroni: false-alarm rate below 1e-4 for
    all of them); entries with P in {0, 1} must be exact. D_hat(J) is a sum of independent binomial
    fractions, and so is Phi_hat(0.3): |D_hat - D| <= 4.5 sd and |Phi_hat - Phi| <= 4.5 sd with the
    exact sd (6.8e-6 per comparison under the normal approximation, at most 32 comparisons).
    Below 3.2e-4 per n, 9.6e-4 for the three.
    """
    samples = 400
    couplings = [0.0, 1.5]
    cps = [n, 2 * n]
    probe = mipt.correlated_erasure(couplings, None, samples, geometry="cycle", checkpoints=cps)
    result = _run(n, 0.1, 0.08, 2 * n, [probe, mipt.tableau_snapshot(checkpoints=cps)])
    js, sizes, values = result.correlated()
    k = result.values["code_dimension"][:, cps]
    prof = ob.correlated_failure(
        values,
        js,
        sizes,
        n,
        k,
        samples=samples,
        sampler=result.spec.correlated_sampler("correlated_erasure"),
    )
    assert prof.complete and prof.sampler["method"] == "exact"
    assert prof.se_exact  # independent samples: the binomial errors are the true ones
    channel = ob.correlated_channel_failure(prof, 0.3)
    assert not channel.interpolated_mass.any()
    ring = [(i, (i + 1) % n) for i in range(n)]
    checks = []
    for row in range(result.num_trajectories):
        for e in range(len(cps)):
            table = oracle.info_table(_joint_oracle(result, row, e).info, n)
            for i, coupling in enumerate(couplings):
                exact = oracle.correlated_failure(table, ring, n, coupling)
                d_exact = oracle.robustness(exact, n, int(k[row, e]))
                variance = max(0.0, sum(p * (1 - p) for p in exact[1:n]) / samples)
                assert (
                    abs(prof.robustness[row, e, i] - d_exact) <= 4.5 * math.sqrt(variance) + 1e-12
                )
                weights = _binomial_weights(n, 0.3)
                phi_exact = sum(b * q for b, q in zip(weights, exact, strict=True))
                phi_var = max(
                    0.0, sum(b**2 * q * (1 - q) for b, q in zip(weights, exact, strict=True))
                )
                assert (
                    abs(channel.value[row, e, i, 0] - phi_exact)
                    <= 4.5 * math.sqrt(phi_var / samples) + 1e-12
                )
                for j, m in enumerate(sizes.tolist()):
                    checks.append((int(round(values[row, e, i, j] * samples)), exact[m]))
    trials = len(checks)
    for successes, p in checks:
        if p < 1e-15 or p > 1 - 1e-15:
            assert successes == round(p * samples)
        else:
            assert _binomial_p_value(successes, samples, p) > 1e-4 / trials


def test_correlated_probe_on_interaction_graphs_is_unbiased_across_trajectories():
    """The chain on Watts-Strogatz graphs: per coupling, the mean over 64 trajectories of
    D_hat(J) - D(J) (exact D from the oracle on each trajectory's own graph) is within 4 standard
    errors of the mean of 0. The chains of different trajectories are independent, so the
    deviations are independent; with at least 40 of them (39 degrees of freedom) the two-sided
    false-alarm probability of |t| > 4 is below 2.8e-4 per coupling, 5.6e-4 for both.
    """
    n, samples = 7, 200
    couplings = [0.0, 1.0]
    probe = mipt.correlated_erasure(couplings, None, samples, thin=2, checkpoints=[2 * n])
    result = _run(
        n,
        0.3,
        0.05,
        2 * n,
        [probe, mipt.tableau_snapshot(checkpoints=[2 * n])],
        graphs_count=16,
        circuits=4,
    )
    js, sizes, values = result.correlated()
    k = result.values["code_dimension"][:, [2 * n]]
    prof = ob.correlated_failure(
        values,
        js,
        sizes,
        n,
        k,
        samples=samples,
        sampler=result.spec.correlated_sampler("correlated_erasure"),
    )
    assert prof.sampler["method"] == "chain"
    assert not prof.se_exact  # the chain: binomial errors are lower bounds
    deviations = {c: [] for c in couplings}
    for row in range(result.num_trajectories):
        if k[row, 0] == 0:
            continue
        edges = result.spec.graphs[result.graph_index[row]].tolist()
        table = oracle.info_table(_joint_oracle(result, row, 0).info, n)
        for i, coupling in enumerate(couplings):
            exact = oracle.correlated_failure(table, edges, n, coupling)
            deviations[coupling].append(
                prof.robustness[row, 0, i] - oracle.robustness(exact, n, int(k[row, 0]))
            )
    for coupling, devs in deviations.items():
        d = np.array(devs)
        assert d.size >= 40, d.size
        if np.all(d == 0):
            continue
        t_stat = d.mean() / (d.std(ddof=1) / math.sqrt(d.size))
        assert abs(t_stat) <= 4.0, (coupling, t_stat)


# ---------------------------------------------------------------------------------------------
# retrospective observables: l(m, J), d_conn, the half-cut entropy (v1.4 §5.3, §5.6, §5.7)
# ---------------------------------------------------------------------------------------------


def _reference_region_seed(key, t, m, coupling, ident, version):
    """The derivation of design §3 (ruling R-SEED) written out: the tag, then the fields in order."""

    def w(text):
        value = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")
        return [value % 2**32, value // 2**32]

    bits = struct.unpack("<Q", struct.pack("<d", coupling))[0]
    return (
        0x5753524F,
        *w(key.experiment_id),
        key.n,
        *w(key.beta.token),
        *w(key.p.token),
        key.g,
        key.c,
        key.r,
        t,
        m,
        bits % 2**32,
        bits // 2**32,
        *w(ident),
        *w(version),
    )


def test_region_seed_derivation():
    base = TrajectoryKey("ws-pilot-001", 16, "0.1", "0.2", 3, 1)
    keys = [
        base,
        TrajectoryKey("ws-pilot-002", 16, "0.1", "0.2", 3, 1),
        TrajectoryKey("ws-pilot-001", 24, "0.1", "0.2", 3, 1),
        TrajectoryKey("ws-pilot-001", 16, "1/7", "0.2", 3, 1),
        TrajectoryKey("ws-pilot-001", 16, "0.1", "0.25", 3, 1),
        TrajectoryKey("ws-pilot-001", 16, "0.1", "0.2", 4, 1),
        TrajectoryKey("ws-pilot-001", 16, "0.1", "0.2", 3, 0),
        TrajectoryKey("ws-pilot-001", 16, "0.1", "0.2", 3, 1, 1),
    ]
    seeds = set()
    for key in keys:
        for t in (0, 5):
            for m in (1, 3):
                for coupling in (0.0, 0.5, -1.0):
                    for version in ("1", "2"):
                        seed = ob.region_seed(key, t, m, coupling, "mean_leakage", version)
                        assert seed == _reference_region_seed(
                            key, t, m, coupling, "mean_leakage", version
                        )
                        assert len(seed) == ob.REGION_SEED_WORDS
                        assert all(0 <= v < 2**32 for v in seed)
                        seeds.add(seed)
    assert len(seeds) == len(keys) * 2 * 2 * 3 * 2  # every field changes the seed
    tags = {seed[0] for seed in seeds}
    assert tags == {ob.REGION_SEED_TAG}
    assert not tags & {0x57534731, 0x57534331}  # the graph (WSG1) and run (WSC1) domains
    assert int.from_bytes(b"WSRO", "big") == ob.REGION_SEED_TAG
    # the identity module's token words and the private string words are one convention
    for token in ("0", "1", "0.1", "1/7", "0.25", "0.0625"):
        assert ob._text_words(token) == identity.token_words(token)
    assert ob.region_seed(str(base), 2, 3, 0.5, "a", "1") == ob.region_seed(
        base, 2, 3, 0.5, "a", "1"
    )
    assert ob.region_seed(base, 2, 3, -0.0, "a", "1") == ob.region_seed(base, 2, 3, 0.0, "a", "1")
    assert ob.region_seed(base, 2, 3, 1, "a", "1") == ob.region_seed(base, 2, 3, 1.0, "a", "1")
    assert ob.region_seed(base, 2, 3, 0.5, "a", "1") != ob.region_seed(base, 2, 3, 0.5, "b", "1")
    # the protocol is deliberately not a field (as in the graph and run seeds): F and P share
    product = dataclasses.replace(base, protocol="P")
    assert product != base and str(product) != str(base)
    assert ob.region_seed(product, 2, 3, 0.5, "a", "1") == ob.region_seed(base, 2, 3, 0.5, "a", "1")
    for args, message in (
        ((base, -1, 0, 0.0, "a", "1"), "t must be at least 0"),
        ((base, 2**32, 0, 0.0, "a", "1"), r"below 2\*\*32"),
        ((base, 0, 0, math.nan, "a", "1"), "J"),
        ((base, 0, 0, 1e7, "a", "1"), "1000000"),
        ((base, 0, 0, True, "a", "1"), "number"),
        ((base, 0, 0, 0.0, "a\x00b", "1"), "NUL"),
        ((base, 0, 0, 0.0, "a", ""), "non-empty"),
        ((base, 0.5, 0, 0.0, "a", "1"), "integer"),
        (("not-a-key", 0, 0, 0.0, "a", "1"), "trajectory key"),
    ):
        with pytest.raises(ValueError, match=message):
            ob.region_seed(*args)
    with pytest.raises(TypeError, match="TrajectoryKey"):
        ob.region_seed(17, 0, 0, 0.0, "a", "1")


def _reference_bootstrap_seed(key, ident, version):
    """The peak-bootstrap seed written out: "WSRB", the key's fields, the id and the version."""

    def w(text):
        value = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")
        return [value % 2**32, value // 2**32]

    return (
        int.from_bytes(b"WSRB", "big"),
        *w(key.experiment_id),
        key.n,
        *w(key.beta.token),
        *w(key.p.token),
        key.g,
        key.c,
        key.r,
        *w(ident),
        *w(version),
    )


def test_bootstrap_seed_derivation_and_the_bootstrap_refuses_fresh_entropy():
    base = TrajectoryKey("ws-pilot-001", 16, "0.1", "0.2", 3, 1)
    keys = [
        base,
        TrajectoryKey("ws-pilot-002", 16, "0.1", "0.2", 3, 1),
        TrajectoryKey("ws-pilot-001", 24, "0.1", "0.2", 3, 1),
        TrajectoryKey("ws-pilot-001", 16, "1/7", "0.2", 3, 1),
        TrajectoryKey("ws-pilot-001", 16, "0.1", "0.25", 3, 1),
        TrajectoryKey("ws-pilot-001", 16, "0.1", "0.2", 4, 1),
        TrajectoryKey("ws-pilot-001", 16, "0.1", "0.2", 3, 0),
        TrajectoryKey("ws-pilot-001", 16, "0.1", "0.2", 3, 1, 1),
    ]
    seeds = set()
    for key in keys:
        for ident in ("peak_D", "peak_D2"):
            for version in ("1", "2"):
                seed = ob.bootstrap_seed(key, ident, version)
                assert seed == _reference_bootstrap_seed(key, ident, version)
                assert len(seed) == ob.BOOTSTRAP_SEED_WORDS == 15
                assert all(0 <= v < 2**32 for v in seed)
                seeds.add(seed)
    assert len(seeds) == len(keys) * 2 * 2  # every field changes the seed
    assert {s[0] for s in seeds} == {ob.BOOTSTRAP_SEED_TAG}
    # distinct from the region (WSRO), graph (WSG1) and run (WSC1) domains in the first word
    assert ob.BOOTSTRAP_SEED_TAG not in {ob.REGION_SEED_TAG, 0x57534731, 0x57534331}
    assert ob.bootstrap_seed(str(base), "peak_D", "1") == ob.bootstrap_seed(base, "peak_D", "1")
    product = dataclasses.replace(base, protocol="P")  # not a field, as in region_seed
    assert ob.bootstrap_seed(product, "peak_D", "1") == ob.bootstrap_seed(base, "peak_D", "1")
    # the bootstrap: a recorded seed reproduces it; fresh entropy (None) and other types are refused
    t = np.array([0, 1, 2, 4])
    first = np.random.default_rng(1).integers(1, 7, size=(4, 30))
    seed = ob.bootstrap_seed(base, "peak_D", "1")
    boot = ob.bootstrap_peak(t, first, 6, z=1.0, replicates=20, seed=seed)
    again = ob.bootstrap_peak(t, first, 6, z=1.0, replicates=20, seed=list(seed))
    np.testing.assert_array_equal(boot.t_star, again.t_star)
    assert boot.seed == again.seed == seed and isinstance(boot.seed, tuple)
    assert ob.bootstrap_peak(t, first, 6, z=1.0, replicates=5, seed=np.int64(3)).seed == 3
    for bad in (None, 1.5, "1", True, [1, -2], [], np.random.SeedSequence(1)):
        with pytest.raises(ValueError, match="seed must"):
            ob.bootstrap_peak(t, first, 6, z=1.0, replicates=5, seed=bad)


def _engine_code(n, beta, p, layers, seed, want_code=True):
    """One stored code with k > 0 (if wanted) and its graph, from a short engine run."""
    result = _run(n, beta, p, layers, [mipt.tableau_snapshot(checkpoints=[layers])], seed=seed)
    for row in range(result.num_trajectories):
        code = result.code(row, 0)
        if (code.num_logical > 0) == want_code:
            return code, result.spec.graphs[result.graph_index[row]], result, row
    raise AssertionError("no such trajectory")


def test_mean_leakage_is_keyed_and_reproducible():
    n = 9
    code, edges, result, row = _engine_code(n, 0.2, 0.05, 18, seed=3)
    g = int(result.graph_index[row])
    c = int(result.trajectories[row]) % result.spec.circuits_per_graph
    key = TrajectoryKey("ws-test", n, "0.2", "0.05", g, c)
    # without replica exchange at J = 2 the chain's records are correlated (ESS < S), so the
    # documented ESS, not the sample count, must set the error
    m, coupling, options = 3, 2.0, {"samples": 60, "tempering": False}
    first = ob.mean_leakage(code, edges, m, coupling, key=key, t=18, **options)
    again = ob.mean_leakage(code, edges, m, coupling, key=str(key), t=18, **options)
    assert (first.value, first.se, first.ess, first.seed) == (
        again.value,
        again.se,
        again.ess,
        again.seed,
    )
    assert first.seed == ob.region_seed(key, 18, m, coupling, "mean_leakage", "1")
    drawn = erasures.sample_regions(
        edges, n, m, [coupling], 60, tempering=False, seed=list(first.seed)
    )
    infos = np.array([code.lost_information(r) for r in drawn.regions[0]], dtype=float)
    assert first.value == infos.mean() and first.method == drawn.method == "chain"
    assert first.ess == min(erasures.chain_diagnostics(infos)["ess"], 60.0) and first.ess < 50
    assert abs(first.se - math.sqrt(infos.var(ddof=1) / first.ess)) < 1e-12
    for other_key, version in ((TrajectoryKey("ws-test", n, "0.2", "0.1", g, c), "1"), (key, "2")):
        other = ob.mean_leakage(
            code, edges, m, coupling, key=other_key, t=18, version=version, **options
        )
        assert other.seed != first.seed
    with pytest.raises(ValueError, match="n = 8"):
        ob.mean_leakage(
            code, edges, m, coupling, key=TrajectoryKey("ws-test", 8, "0", "0.1", 0, 0), t=1
        )
    # exact zero and full cases
    k = code.num_logical
    assert ob.mean_leakage(code, edges, 0, coupling, key=key, t=18).value == 0.0
    full = ob.mean_leakage(code, edges, n, coupling, key=key, t=18, samples=10)
    assert full.value == 2 * k and full.se == 0.0
    purified, edges0, _, _ = _engine_code(n, 0.2, 0.6, 18, seed=3, want_code=False)
    nothing = ob.mean_leakage(purified, edges0, 4, coupling, key=key, t=18)
    assert (nothing.value, nothing.se, nothing.method, nothing.samples) == (0.0, 0.0, "none", 0)
    # through the registry, with a stored snapshot
    snap = ob.SnapshotInput(num_system=n, t=18, key=key, code=code, edges=edges)
    value = ob.compute(
        "mean_leakage", snap, {"m": m, "J": coupling, "samples": 60, "tempering": False}
    )
    assert (value.value, value.se, value.method) == (first.value, first.se, "chain")
    assert value.data["seed"] == list(first.seed) and value.data["samples"] == 60
    exact = ob.compute("mean_leakage", snap, {"m": m, "J": coupling, "route": "enumerate"})
    assert exact.method == "enumerate" and exact.se == 0.0
    assert exact.value == ob.mean_leakage_exact(code, edges, m, [coupling])[0]
    with pytest.raises(ValueError, match=r"size must be in \[0, 9\]"):
        ob.mean_leakage(code, edges, 10, 0.0, key=key, t=0)


def test_a_registered_version_keeps_its_own_region_seeds():
    """A changed setting becomes a new version (design §3, §5), registered beside the old one.
    Values of version 1 must keep their seeds and values; version 2 and a renamed copy draw
    their own regions (Proposition 18(c))."""
    n, t = 8, 16
    code, edges, _, _ = _engine_code(n, 0.3, 0.05, t, seed=5)
    key = TrajectoryKey("ws-test", n, "0.3", "0.05", 0, 1)
    snap = ob.SnapshotInput(num_system=n, t=t, key=key, code=code, edges=edges)
    params = {"m": 3, "J": 1.0, "samples": 40}
    before = ob.compute("mean_leakage", snap, params, version="1")
    assert before.data["seed"] == list(ob.region_seed(key, t, 3, 1.0, "mean_leakage", "1"))
    assert (before.observable_id, before.version) == ("mean_leakage", "1")
    assert before.params_key == ob.params_key("mean_leakage", params)
    v1 = ob.get("mean_leakage", "1")
    v2 = dataclasses.replace(v1, version="2", definition=v1.definition + " (a new setting)")
    copy = dataclasses.replace(v1, id="mean_leakage_copy")
    ob.register(v2)
    ob.register(copy)
    try:
        assert ob.get("mean_leakage") is v2  # the latest registered
        after = ob.compute("mean_leakage", snap, params, version="1")
        assert after.data["seed"] == before.data["seed"]
        assert (after.value, after.se, after.version) == (before.value, before.se, "1")
        latest = ob.compute("mean_leakage", snap, params)  # no version: the latest, and it says so
        assert latest.version == "2"
        assert latest.data["seed"] == list(ob.region_seed(key, t, 3, 1.0, "mean_leakage", "2"))
        renamed = ob.compute("mean_leakage_copy", snap, params)
        assert renamed.observable_id == "mean_leakage_copy"
        assert renamed.data["seed"] == list(
            ob.region_seed(key, t, 3, 1.0, "mean_leakage_copy", "1")
        )
        seeds = {tuple(v.data["seed"]) for v in (before, latest, renamed)}
        assert len(seeds) == 3
    finally:
        del ob._REGISTRY[("mean_leakage", "2")]
        del ob._REGISTRY[("mean_leakage_copy", "1")]
    assert ob.get("mean_leakage") is v1


def test_mean_leakage_caps_the_chain_ess_at_the_sample_count(monkeypatch):
    """Sokal's window can estimate tau_int < 1 (ESS > S) for a fast-mixing chain; the error
    must then be the independent one, s / sqrt(S), never smaller (Proposition 17). Two real
    cases: the swap chain at J = 0, which has one rung and so no replica exchange (untempered;
    raw ESS 106.7 at S = 60), and a tempered chain at J = 0.5 (replica exchange on; raw ESS
    91.8); then a diagnostic forced to report a huge ESS."""
    n, m, samples = 9, 3, 60
    code, edges, _, _ = _engine_code(n, 0.2, 0.05, 18, seed=3)
    key = TrajectoryKey("ws-test", n, "0.2", "0.05", 0, 0)
    estimates = {}
    for coupling, tempered in ((0.0, False), (0.5, True)):
        seed = ob.region_seed(key, 18, m, coupling, "mean_leakage", "1")
        drawn = erasures.sample_regions(edges, n, m, [coupling], samples, seed=list(seed))
        infos = np.array([code.lost_information(r) for r in drawn.regions[0]], dtype=float)
        raw = erasures.chain_diagnostics(infos)["ess"]
        # the case: the uncapped ESS exceeds S (asserted, so a changed stream fails loudly)
        assert drawn.method == "chain" and drawn.tempering is tempered and raw > samples
        est = ob.mean_leakage(code, edges, m, coupling, key=key, t=18, samples=samples)
        assert est.method == "chain" and est.tempering is tempered and est.value == infos.mean()
        assert est.ess == samples
        assert abs(est.se - math.sqrt(infos.var(ddof=1) / samples)) < 1e-12
        estimates[coupling] = est
    # whatever the diagnostic reports, the cap holds
    real = erasures.chain_diagnostics
    monkeypatch.setattr(erasures, "chain_diagnostics", lambda series: {**real(series), "ess": 1e9})
    for coupling, est in estimates.items():
        capped = ob.mean_leakage(code, edges, m, coupling, key=key, t=18, samples=samples)
        assert capped.ess == samples and capped.se == est.se


def test_mean_leakage_matches_the_exact_law_with_exact_samples(rng):
    """On the ring (exact independent samples): |l_hat - l| <= 4.5 sd with the exact
    sd = sqrt(Var_Q(I) / S) from the oracle's law; S = 400, so l_hat is close to normal and each
    comparison has a false-alarm probability of about 7e-6 (below 1e-4 for the 12)."""
    samples = 400
    compared = 0
    for n in (6, 8):
        ring = [(i, (i + 1) % n) for i in range(n)]
        while compared < (6 if n == 6 else 12):
            measured = rng.choice(n, size=int(rng.integers(0, 3)), replace=False)
            joint = oracle.Joint(oracle.random_joint_state(n, rng, measured), n)
            if joint.k == 0:
                continue
            table = oracle.info_table(joint.info, n)
            code = StabilizerCode(
                oracle.ints_to_dense(joint.physical_generators(), n), num_qubits=n
            )
            m = int(rng.integers(1, n))
            coupling = float(rng.choice([0.0, 0.8, 2.0]))
            law = oracle.correlated_law(ring, n, m, coupling)
            mean = sum(q * table[a] for a, q in law.items())
            variance = sum(q * (table[a] - mean) ** 2 for a, q in law.items())
            key = TrajectoryKey("ws-test", n, "0", "0.1", 0, compared)
            est = ob.mean_leakage(code, ring, m, coupling, key=key, t=1, samples=samples)
            assert est.method == "exact" and est.ess == samples
            assert abs(est.value - mean) <= 4.5 * math.sqrt(variance / samples) + 1e-12
            compared += 1


def test_mean_leakage_with_the_chain_is_unbiased():
    """The chain on a Watts-Strogatz graph: 48 independent seeds (circuit labels 0..47) of one
    code; the mean of l_hat - l is within 4 standard errors of 0 (t with 47 degrees of freedom:
    two-sided false-alarm probability 2.3e-4 per case, below 5e-4 for the two)."""
    n = 8
    code, edges, _, _ = _engine_code(n, 0.3, 0.05, 16, seed=5)
    joint_table = oracle.Code(oracle.dense_to_ints(code.stabilizer_matrix()), n)
    table = {mask: joint_table.lost(mask) for mask in range(1 << n)}
    keys = [TrajectoryKey("ws-test", n, "0.3", "0.05", 0, c) for c in range(48)]
    for m, coupling in ((3, 0.0), (4, 1.2)):
        exact = oracle.mean_leakage(table, edges.tolist(), n, m, coupling)
        devs = np.array(
            [
                ob.mean_leakage(code, edges, m, coupling, key=keys[c], t=16, samples=150).value
                - exact
                for c in range(48)
            ]
        )
        if np.all(devs == 0):
            continue
        t_stat = devs.mean() / (devs.std(ddof=1) / math.sqrt(devs.size))
        assert abs(t_stat) <= 4.0, (m, coupling, t_stat)


def test_connected_distance_against_the_oracle(rng):
    cases = 0
    for n in range(2, 9):
        candidates = [graphs.cycle(n) if n >= 3 else graphs.path(n), graphs.path(n)]
        if n >= 5:
            candidates += [
                graphs.watts_strogatz(n, 4, beta, seed=(1, n)) for beta in (0.0, 0.5, 1.0)
            ]
        sparse = [(a, b) for a, b in ((0, 1), (2, 3), (1, 4), (5, 6), (3, 7)) if b < n]
        candidates.append(
            np.array(sparse if sparse else [(0, 1)], dtype=np.int64)
        )  # often disconnected
        for edges in candidates:
            measured = rng.choice(n, size=int(rng.integers(0, n)), replace=False)
            joint = oracle.Joint(oracle.random_joint_state(n, rng, measured), n)
            table = oracle.info_table(joint.info, n)
            code = StabilizerCode(
                oracle.ints_to_dense(joint.physical_generators(), n), num_qubits=n
            )
            expected = oracle.connected_distance(
                table, graphs.from_edges(edges, n).tolist(), n, joint.k
            )
            assert ob.connected_distance(code, edges) == expected
            cases += 1
    assert cases >= 30
    big = StabilizerCode(["Z" * 21])
    with pytest.raises(ValueError, match="max_n = 20"):
        ob.connected_distance(big, graphs.cycle(21))
    assert (
        ob.connected_distance(big, graphs.cycle(21), max_n=21) == 1.0
    )  # Z_q is a logical operator


def test_connected_and_cyclic_distances_can_differ():
    # the [[4,2,2]] code on the even qubits of 8, Z on the odd ones: {0, 2} fails and is connected
    # in C(8, 2), but every interval of length 2 holds one code qubit
    code = StabilizerCode(["X_X_X_X_", "Z_Z_Z_Z_", "_Z______", "___Z____", "_____Z__", "_______Z"])
    ring = graphs.watts_strogatz(8, 4, 0.0, seed=0)  # C(8, 2)
    assert ob.cyclic_distance_from_code(code) == 3.0
    assert ob.connected_distance(code, ring) == 2.0
    by_code = oracle.Code(oracle.dense_to_ints(code.stabilizer_matrix()), 8)
    table = {mask: by_code.lost(mask) for mask in range(256)}
    assert oracle.cyclic_distance(table, 8, 2) == 3.0
    assert oracle.connected_distance(table, ring.tolist(), 8, 2) == 2.0
    # the [[4,2,2]] code on {0, 1, 4, 5} with the edges {0,1} and {4,5} rewired to {0,3} and {4,7}:
    # the interval {0, 1} fails, but no two code qubits are adjacent any more
    code = StabilizerCode(["XX__XX__", "ZZ__ZZ__", "__Z_____", "___Z____", "______Z_", "_______Z"])
    rewired = [tuple(e) for e in ring.tolist() if tuple(e) not in {(0, 1), (4, 5)}] + [
        (0, 3),
        (4, 7),
    ]
    assert ob.cyclic_distance_from_code(code) == 2.0
    assert ob.connected_distance(code, rewired) == 3.0
    assert ob.connected_distance(StabilizerCode(["ZZ", "XX"]), graphs.path(2)) == 0.0  # k = 0


def test_cyclic_distance_wraps_around_the_ring():
    """The [[4,2,2]] code on {0, 2, 5, 7} of 8 qubits: the interval {7, 0} wraps and fails, so
    d_cyc = 2; read as an open chain the first failing window would have length 3."""
    code = StabilizerCode(["X_X__X_X", "Z_Z__Z_Z", "_Z______", "___Z____", "____Z___", "______Z_"])
    assert ob.cyclic_distance_from_code(code) == 2.0
    assert code.contiguous_distance(periodic=False) == 3
    by_code = oracle.Code(oracle.dense_to_ints(code.stabilizer_matrix()), 8)
    table = {mask: by_code.lost(mask) for mask in range(256)}
    assert oracle.cyclic_distance(table, 8, by_code.k) == 2.0


def test_connected_distance_is_at_most_cyclic_on_the_unrewired_graph():
    """On C(n, 2) every interval is connected, so d_conn <= d_cyc (Proposition 12)."""
    seen = 0
    for n, p in ((8, 0.05), (10, 0.05), (11, 0.08)):
        cps = [1, n // 2, n, 2 * n]
        result = _run(
            n, 0.0, p, 2 * n, [mipt.tableau_snapshot(checkpoints=cps)], graphs_count=1, circuits=4
        )
        edges = result.spec.graphs[0]
        for row in range(result.num_trajectories):
            for e in range(len(cps)):
                code = result.code(row, e)
                if code.num_logical == 0:
                    continue
                assert ob.connected_distance(code, edges) <= ob.cyclic_distance_from_code(code)
                seen += 1
    assert seen >= 20


@pytest.mark.parametrize("n, beta", [(5, 0.0), (6, 0.2), (7, 0.5), (12, 0.1)])
def test_half_cut_entropy_matches_the_engine_probe_and_the_oracle(n, beta):
    cps = [0, 1, 2, n, 2 * n]
    result = _run(
        n,
        beta,
        0.1,
        2 * n,
        [mipt.half_cut_entropy(checkpoints=cps), mipt.tableau_snapshot(checkpoints=cps)],
    )
    probe = result.values["half_cut_entropy"]
    for row in range(result.num_trajectories):
        for e, t in enumerate(cps):
            tableau = result.tableau(row, e)
            value = ob.half_cut_entropy(tableau, n)
            assert value == probe[row, e]
            snap = ob.SnapshotInput(num_system=n, t=t, tableau=tableau)
            assert ob.compute("half_cut_entropy", snap).value == value
            if n <= 7:
                joint = _joint_oracle(result, row, e)
                assert value == joint.entropy((1 << (n // 2)) - 1)


def test_compute_from_stored_snapshots_matches_the_engine_probes():
    n, layers = 9, 18
    cps = [0, 1, 4, 9, 18]
    result = _run(
        n,
        0.2,
        0.1,
        layers,
        [mipt.contiguous_distance(checkpoints=cps), mipt.tableau_snapshot(checkpoints=cps)],
    )
    for row in range(result.num_trajectories):
        edges = result.spec.graphs[result.graph_index[row]]
        for e, t in enumerate(cps):
            snap = ob.SnapshotInput(num_system=n, t=t, tableau=result.tableau(row, e), edges=edges)
            k = result.values["code_dimension"][row, t]
            assert ob.compute("k", snap).value == k
            assert ob.compute("code_rate", snap).value == k / n
            assert ob.compute("stabilizer_rank", snap).value == n - k
            d = ob.compute("d_cyc", snap)
            raw = result.values["contiguous_distance"][row, e]
            assert (math.isnan(raw) and math.isnan(d.value)) or d.value == raw
            assert d.data["has_code"] == (k > 0) and d.data["zero_extended"] == (
                0 if k == 0 else raw
            )
            conn = ob.compute("d_conn", snap)
            assert conn.value == ob.connected_distance(snap.physical_code(), edges)
            assert conn.method == "enumeration" and conn.se == 0.0


def _key(n, protocol="F"):
    return TrajectoryKey("ws-test", n, "0.2", "0.1", 0, 0, 0, protocol)


def test_a_snapshot_is_checked_against_its_n():
    """A stored code or tableau paired with another n is refused, not computed into plausible
    wrong values (k, rate, rank, half-cut entropy all depend on n). With the trajectory's key
    the joint width is exact for its protocol (2n for F, n for P); without one, any width a
    joint state of n system qubits can have is accepted."""
    n = 6
    result = _run(n, 0.2, 0.1, n, [mipt.tableau_snapshot(checkpoints=[n])], graphs_count=1)
    tableau, code = result.tableau(0, 0), result.code(0, 0)
    assert tableau.num_qubits == 2 * n and code.num_qubits == n
    keyed = ob.SnapshotInput(n, n, key=_key(n), tableau=tableau)
    assert ob.compute("k", keyed).value == code.num_logical
    assert ob.compute("k", ob.SnapshotInput(n, n, tableau=tableau)).value == code.num_logical
    # with a key: an F snapshot of n = 6 under any other n, or under a P key, is refused (also
    # n' = 12 and 11, where its 12 qubits are n' and n' + 1, widths the unkeyed check accepts)
    for wrong in (4, 5, 7, 8, 11, 12):
        with pytest.raises(
            ValueError, match=f"protocol-F snapshot of n = {wrong} .* has {2 * wrong}"
        ):
            ob.SnapshotInput(wrong, n, key=_key(wrong), tableau=tableau)
    for wrong in (6, 11):
        with pytest.raises(ValueError, match=f"protocol-P snapshot of n = {wrong} .* has {wrong}"):
            ob.SnapshotInput(wrong, n, key=_key(wrong, "P"), tableau=tableau)
    # what a width cannot see, one way: this F state of n = 6 has the width of a P state of 12
    ob.SnapshotInput(2 * n, n, key=_key(2 * n, "P"), tableau=tableau)
    # without a key: n, n + 1 or 2n
    for wrong in (4, 5, 7, 8):  # 12 qubits are 2n only at n = 6, n + 1 at 11 and n at 12
        with pytest.raises(ValueError, match="the joint tableau has 12 qubits"):
            ob.SnapshotInput(wrong, n, tableau=tableau)
    for other in (11, 12):  # one reference, no reference: widths a joint state can have
        ob.SnapshotInput(other, n, tableau=tableau)
    # a P snapshot (the pure n-qubit state): k = 0; refused under an F key of its n or of n - 1
    m = 8
    pure = mipt.MonitoredCircuit(
        num_system=m,
        edges=[_edges(m, 0.2, 0)],
        layers=4,
        measurement_rate=0.1,
        reference="none",
        schedule="random_edges",
        checkpoints=[4],
        probes=[mipt.tableau_snapshot(checkpoints=[4])],
    ).run(seed=3)
    state = pure.tableau(0, 0)
    assert state.num_qubits == m
    p_snap = ob.SnapshotInput(m, 4, key=_key(m, "P"), tableau=state)
    assert ob.compute("k", p_snap).value == 0.0
    for wrong in (m, m - 1):
        with pytest.raises(ValueError, match=f"protocol-F snapshot of n = {wrong}"):
            ob.SnapshotInput(wrong, 4, key=_key(wrong), tableau=state)
    # and the other way: this P state of n = 8 has the width of an F state of n = 4, so under the
    # F key of 4 it is computed as one, into a plausible wrong k (its own k is 0, above)
    wrong_f = ob.SnapshotInput(m // 2, 4, key=_key(m // 2), tableau=state)
    wrong_k = state.physical_code(m // 2).num_logical
    assert ob.compute("k", wrong_f).value == wrong_k == 2  # doc 04 §5.9 quotes it
    # the code: exactly n qubits, with or without a key
    for wrong in (5, 8):
        with pytest.raises(ValueError, match=f"the code has 6 qubits, not num_system = {wrong}"):
            ob.SnapshotInput(wrong, n, code=code)
        with pytest.raises(ValueError, match=f"the code has 6 qubits, not num_system = {wrong}"):
            ob.SnapshotInput(wrong, n, key=_key(wrong), code=code)
    # the graph: endpoints in 0..n-1, checked when the snapshot is built and normalized once
    with pytest.raises(ValueError, match=r"edges must be .*\(endpoints 0\.\.5\): edge endpoint 7"):
        ob.SnapshotInput(n, n, code=code, edges=graphs.cycle(8))
    with pytest.raises(ValueError, match="self-loops"):
        ob.SnapshotInput(n, n, code=code, edges=[(0, 1), (2, 2)])
    snap = ob.SnapshotInput(n, n, code=code, edges=[(1, 0), (2, 1), (0, 1)])
    assert snap.edges.tolist() == [[0, 1], [1, 2]] and snap.graph().tolist() == [[0, 1], [1, 2]]
    with pytest.raises(TypeError, match="StabilizerCode"):
        ob.SnapshotInput(n, n, code="XXXXXX")
    with pytest.raises(TypeError, match="PhaseFreeTableau"):
        ob.SnapshotInput(n, n, tableau=code)


# ---------------------------------------------------------------------------------------------
# peaks (v1.4 §5.8; validation requirement 13)
# ---------------------------------------------------------------------------------------------


def test_exact_peaks_ties_log_midpoint_and_horizon():
    t = [0, 1, 2, 4, 8, 16]
    p = ob.peak(t, [0, 1, 3, 3, 2, 0], rule="exact", k_at=[6, 5, 4, 3, 1, 0])
    assert p.d_star == 3.0 and p.maximizing_set.tolist() == [2, 4]
    assert p.t_star == pytest.approx(math.sqrt(8), rel=1e-15) and p.first_argmax == 2
    assert not p.missing and not p.at_horizon and p.horizon == 16
    assert p.k_on_set.tolist() == [4, 3] and p.rule == "exact" and p.z is None
    assert not p.t_star_is_checkpoint and p.to_dict()["t_star_is_checkpoint"] is False
    edge = ob.peak(t, [0, 1, 2, 3, 4, 4], rule="exact")
    assert edge.maximizing_set.tolist() == [8, 16] and edge.at_horizon
    assert edge.t_star == pytest.approx(math.sqrt(128), rel=1e-15)
    single = ob.peak(t, [99, 1, 2, 5, 4, 0], rule="exact")  # the entry at t = 0 is ignored
    assert single.t_star == 4.0 and single.d_star == 5.0 and single.t_star_is_checkpoint
    spread = ob.peak(t, [0, 1, 0, 5, 0, 5], rule="exact")  # A = {4, 16}: T* = 8, a checkpoint
    assert spread.t_star == pytest.approx(8.0, rel=1e-15) and spread.t_star_is_checkpoint
    assert spread.first_argmax == 4
    beyond = ob.peak(t, [0, 1, 2, 3, 4, 4], rule="exact", horizon=32)
    assert not beyond.at_horizon and beyond.horizon == 32
    assert ob.peak(t, [0, 1, 3, 3, 2, 0], rule="exact").to_dict()["maximizing_set"] == [2, 4]


def test_peaks_missing_and_invalid():
    t = [0, 1, 2, 4]
    zero = ob.peak(t, [3, 0, 0, 0], rule="exact")
    assert (
        zero.missing and "d* = 0" in zero.reason and math.isnan(zero.t_star) and not zero.at_horizon
    )
    assert zero.maximizing_set.tolist() == [1, 2, 4] and zero.to_dict()["t_star"] is None
    for values, word in (
        ([0, 1, math.nan, 2], "zero-extended"),
        ([0, 1, -1, 2], "nonnegative"),
        ([0, 1, math.inf, 2], "finite"),
    ):
        bad = ob.peak(t, values, rule="exact", k_at=[3, 2, 1, 0])
        assert bad.missing and word in bad.reason and bad.maximizing_set.size == 0
        assert bad.k_on_set.size == 0 and bad.first_argmax == -1 and math.isnan(bad.d_star)
    nothing = ob.peak([0], [5], rule="exact")
    assert nothing.missing and nothing.reason == "no positive checkpoint"
    noisy = ob.peak(t, [0, 1, 2, 3], [0, 0.1, math.nan, 0.1], rule="plateau", z=1)
    assert noisy.missing and "se(2)" in noisy.reason
    for kwargs, message in (
        ({"rule": "best"}, "rule"),
        ({"rule": "plateau"}, "needs z"),
        ({"rule": "plateau", "z": 1}, "needs se"),
        ({"rule": "plateau", "z": -1, "se": [0, 1, 1, 1]}, "z must be"),
        ({"rule": "exact", "horizon": 2}, "below the last"),
        ({"rule": "exact", "k_at": [1, 2]}, "k_at"),
        ({"rule": "plateau", "z": 1, "se": [1, 1]}, "se must"),
    ):
        with pytest.raises(ValueError, match=message):
            ob.peak(t, [0, 1, 2, 3], **kwargs)
    with pytest.raises(ValueError, match="one value per checkpoint"):
        ob.peak(t, [0, 1, 2], rule="exact")
    with pytest.raises(ValueError, match="increase strictly"):
        ob.peak([0, 2, 1], [0, 1, 2], rule="exact")


def test_plateau_rule_hand_cases_and_nesting(rng):
    t = [1, 2, 4, 8]
    d = [1.0, 2.0, 1.9, 1.0]
    se = [0.05] * 4
    assert ob.peak(t, d, se, rule="plateau", z=0).maximizing_set.tolist() == [2]
    assert ob.peak(t, d, se, rule="plateau", z=1).maximizing_set.tolist() == [2]  # 2 - 0.0707 > 1.9
    wide = ob.peak(t, d, se, rule="plateau", z=2)  # 2 - 0.1414 < 1.9
    assert wide.maximizing_set.tolist() == [2, 4] and wide.t_star == pytest.approx(math.sqrt(8))
    many = ob.peak_sensitivity(t, d, se, [0, 1, 2])
    assert [p.maximizing_set.tolist() for p in many] == [[2], [2], [2, 4]]
    assert [p.z for p in many] == [0.0, 1.0, 2.0]
    with pytest.raises(ValueError, match="at least one"):
        ob.peak_sensitivity(t, d, se, [])
    for _ in range(200):
        size = int(rng.integers(2, 10))
        times = np.cumsum(rng.integers(1, 5, size=size))
        values = rng.normal(5, 1, size=size).clip(0)
        errors = rng.uniform(0, 0.5, size=size)
        sets = [
            set(ob.peak(times, values, errors, rule="plateau", z=z).maximizing_set.tolist())
            for z in (0, 0.5, 1, 2, 4)
        ]
        assert all(a <= b for a, b in zip(sets, sets[1:], strict=False))
        assert sets[0] == set(ob.peak(times, values, rule="exact").maximizing_set.tolist())


def _reference_plateau(t, d, se, z):
    """The plateau rule of design §5 written out with Python floats: over t > 0, d* the maximum,
    t0 the first maximizer, A = {t : d(t) >= d* - z sqrt(se(t)^2 + se(t0)^2)}, and
    T* = exp(mean of ln t over A) (a single checkpoint exactly; NaN when d* = 0)."""
    points = [(int(a), float(b), float(c)) for a, b, c in zip(t, d, se, strict=True) if a > 0]
    d_star = max(b for _, b, _ in points)
    s_first = next(c for _, b, c in points if b == d_star)
    chosen = [a for a, b, c in points if b >= d_star - z * math.sqrt(c * c + s_first * s_first)]
    if d_star == 0:
        return chosen, math.nan
    if len(chosen) == 1:
        return chosen, float(chosen[0])
    return chosen, math.exp(sum(math.log(a) for a in chosen) / len(chosen))


def _same_t_star(a, b):
    return (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, rel=1e-12)


def test_plateau_rule_uses_the_first_maximizers_error():
    """Unequal errors, so that membership depends on se(t0), the first maximizer's error."""
    # (a) a checkpoint at t = 0, and t0 = 2 is not the smallest positive depth: t = 1 stays in,
    # 5.0 >= 5.3 - sqrt(0.01^2 + 0.3^2) = 4.99983 (with se(1) in place of se(t0) it would not)
    t, d, se = [0, 1, 2, 4], [0, 5.0, 5.3, 5.2], [0, 0.01, 0.3, 0.3]
    a = ob.peak(t, d, se, rule="plateau", z=1)
    assert a.maximizing_set.tolist() == [1, 2, 4] and a.first_argmax == 2
    assert a.t_star == pytest.approx(2.0, rel=1e-15)
    assert _reference_plateau(t, d, se, 1) == ([1, 2, 4], a.t_star)
    # (b) t0 = 4 with se 0.02: t = 2 is out, 3.7 < 4.2 - sqrt(0.05^2 + 0.02^2) = 4.146 (with the
    # last checkpoint's error 0.5 in place of se(t0) it would be in: 3.7 >= 4.2 - 0.5025 = 3.6975)
    t, d, se = [1, 2, 4, 8], [4.0, 3.7, 4.2, 3.9], [0.25, 0.05, 0.02, 0.5]
    b = ob.peak(t, d, se, rule="plateau", z=1)
    assert b.maximizing_set.tolist() == [1, 4, 8]
    assert b.t_star == pytest.approx(32 ** (1 / 3), rel=1e-15)
    assert _reference_plateau(t, d, se, 1)[0] == [1, 4, 8]


def test_plateau_rule_matches_its_definition_on_random_traces(rng):
    """400 random traces (some with t = 0, some with exact ties), unequal errors, several z:
    the set and T* equal the rule written out (_reference_plateau)."""
    for _ in range(400):
        times = np.cumsum(rng.integers(1, 5, size=int(rng.integers(1, 10))))
        if rng.random() < 0.5:
            times = np.concatenate([[0], times])
        values = rng.normal(5, 1, size=times.size).clip(0)
        if rng.random() < 0.3:
            values = np.round(values)  # exact ties
        errors = rng.uniform(0, 0.6, size=times.size)
        z = float(rng.choice([0.0, 0.5, 1.0, 2.0, 3.0]))
        got = ob.peak(times, values, errors, rule="plateau", z=z)
        chosen, t_star = _reference_plateau(times, values, errors, z)
        assert got.maximizing_set.tolist() == chosen
        assert _same_t_star(got.t_star, t_star)


def test_peaks_record_whether_their_errors_are_true():
    t, d, se = [1, 2, 4], [1.0, 2.0, 1.5], [0.1, 0.1, 0.1]
    assert ob.peak(t, d, se, rule="plateau", z=1).se_exact is None  # not stated
    chain = ob.peak(t, d, se, rule="plateau", z=1, se_exact=False)
    assert chain.se_exact is False and chain.to_dict()["se_exact"] is False
    assert ob.peak(t, d, se, rule="plateau", z=1, se_exact=True).to_dict()["se_exact"] is True
    assert ob.peak(t, d, rule="exact", se_exact=True).se_exact is None  # no errors are used
    invalid = ob.peak(t, [1.0, math.nan, 1.0], se, rule="plateau", z=1, se_exact=False)
    assert invalid.missing and invalid.se_exact is False
    zero = ob.peak(t, [0.0, 0.0, 0.0], se, rule="plateau", z=1, se_exact=False)
    assert zero.missing and zero.se_exact is False
    swept = ob.peak_sensitivity(t, d, se, [0, 1, 2], se_exact=False)
    assert [p.se_exact for p in swept] == [False] * 3
    with pytest.raises(ValueError, match="se_exact"):
        ob.peak(t, d, se, rule="plateau", z=1, se_exact="no")
    # the bootstrap of D(0): errors from independent orderings are the true ones
    boot = ob.bootstrap_peak(t, [[2, 3, 2], [3, 4, 4], [3, 3, 2]], 5, z=1.0, replicates=3, seed=1)
    assert boot.estimate.se_exact is True


def test_plateau_rule_keeps_a_true_plateau_with_the_stated_probability(rng):
    """True trace 1, 2, 3, 3, 3, 2, 1 with N(0, 0.1^2) noise and z = 2.

    A plateau checkpoint j is left out only if some i has d_i - d_j > z sqrt(2) sigma; for two
    plateau points that has probability Phi_N(-z) each (Phi_N the standard normal distribution
    function), and the points one unit below are 7 sigma away. So P(some plateau point left out)
    <= 6 Phi_N(-2) = 0.1365. The errors given are the true ones here. Over 4000 trials the observed
    fraction must stay below 0.1365 + 4 sqrt(0.1365 x 0.8635 / 4000) = 0.158 (false alarm below 4e-5,
    normal approximation of the binomial).
    """
    t = np.array([1, 2, 4, 8, 16, 32, 64])
    truth = np.array([1, 2, 3, 3, 3, 2, 1], dtype=float)
    sigma, trials = 0.1, 4000
    se = np.full(t.size, sigma)
    left_out = 0
    single = 0
    for _ in range(trials):
        noisy = truth + rng.normal(0, sigma, size=t.size)
        kept = set(ob.peak(t, noisy, se, rule="plateau", z=2).maximizing_set.tolist())
        left_out += not {4, 8, 16} <= kept
        single += ob.peak(t, noisy, se, rule="plateau", z=0).maximizing_set.size == 1
    bound = 6 * 0.5 * math.erfc(2 / math.sqrt(2))
    assert left_out / trials <= bound + 4 * math.sqrt(bound * (1 - bound) / trials)
    assert single == trials  # continuous noise: no ties at z = 0


def test_bootstrap_peak_reproduces_its_documented_draws():
    n, samples, replicates, seed = 6, 40, 30, [11, 2]
    rng = np.random.default_rng(5)
    t = np.array([0, 1, 2, 4, 8])
    first = np.stack([np.sort(rng.integers(1, n + 1, size=samples)) for _ in t])
    boot = ob.bootstrap_peak(t, first, n, z=1.0, replicates=replicates, seed=seed)
    again = ob.bootstrap_peak(t, first, n, z=1.0, replicates=replicates, seed=seed)
    np.testing.assert_array_equal(boot.t_star, again.t_star)
    # the documented draws, written out: PCG64(SeedSequence(seed)), top 53 bits, floor(u S)
    raw = np.random.PCG64(np.random.SeedSequence(seed)).random_raw(replicates * t.size * samples)
    picks = np.floor((raw >> np.uint64(11)).astype(np.float64) * 2.0**-53 * samples).astype(int)
    picks = picks.reshape(replicates, t.size, samples)
    for b in range(replicates):
        values, errors = [], []
        for e in range(t.size):
            draw = first[e][picks[b, e]]
            values.append(draw.mean())
            errors.append(draw.std(ddof=1) / math.sqrt(samples))
        _, reference = _reference_plateau(t, values, errors, 1.0)
        assert _same_t_star(boot.t_star[b], reference)
    point = [_reference_robustness(first[e].tolist(), n) for e in range(t.size)]
    expected, _ = _reference_plateau(t, [v for v, _ in point], [s for _, s in point], 1.0)
    assert boot.estimate.maximizing_set.tolist() == expected
    finite = boot.t_star[~np.isnan(boot.t_star)]
    assert np.all((finite >= 1) & (finite <= 8))
    lo, hi = boot.interval(0.9)
    assert lo <= hi and boot.missing_fraction == float(np.isnan(boot.t_star).mean())
    with pytest.raises(ValueError, match="level"):
        boot.interval(1.0)
    other = ob.bootstrap_peak(t, first, n, z=1.0, replicates=replicates, seed=[11, 3])
    assert not np.array_equal(other.t_star, boot.t_star)


def test_bootstrap_peak_keeps_the_exact_counts_in_every_replicate():
    """Noisy orderings with exhaustive counts up to w = 2 (nonzero, so they never contradict the
    samples). Each replicate's D(0) and error are rebuilt from the documented draws with the
    hybrid estimator written out (_reference_robustness: Proposition 13) and peaked with the rule
    written out (_reference_plateau). Replicates that dropped the counts would use the plain
    estimator and its error; the test checks that this changes T* in some replicates, so it can
    tell the two apart."""
    n, samples, replicates = 8, 30, 60
    rng = np.random.default_rng(12)
    t = np.array([0, 1, 2, 4, 8, 16])
    first = rng.integers(1, n + 1, size=(t.size, samples))
    counts = np.array([[rng.integers(1, 9), rng.integers(1, 29)] for _ in t])
    seed = ob.bootstrap_seed(TrajectoryKey("ws-test", n, "0.1", "0.1", 0, 0), "peak_D", "1")
    boot = ob.bootstrap_peak(
        t, first, n, z=1.0, replicates=replicates, seed=seed, exact_counts=counts
    )
    raw = np.random.PCG64(np.random.SeedSequence(seed)).random_raw(replicates * t.size * samples)
    picks = np.floor((raw >> np.uint64(11)).astype(np.float64) * 2.0**-53 * samples).astype(int)
    picks = picks.reshape(replicates, t.size, samples)
    differs = 0
    for b in range(replicates):
        hybrid, plain = [], []
        for e in range(t.size):
            draw = first[e][picks[b, e]].tolist()
            hybrid.append(_reference_robustness(draw, n, counts[e].tolist()))
            plain.append(_reference_robustness(draw, n))
        _, t_hybrid = _reference_plateau(t, *zip(*hybrid, strict=True), 1.0)
        _, t_plain = _reference_plateau(t, *zip(*plain, strict=True), 1.0)
        assert _same_t_star(boot.t_star[b], t_hybrid), b
        differs += not _same_t_star(t_plain, t_hybrid)
    assert differs >= 5, differs
    point = [_reference_robustness(first[e].tolist(), n, counts[e].tolist()) for e in range(t.size)]
    assert (
        boot.estimate.maximizing_set.tolist()
        == _reference_plateau(t, *zip(*point, strict=True), 1.0)[0]
    )


def test_bootstrap_peak_of_noiseless_samples_is_the_estimate():
    t = np.array([1, 2, 4, 8])
    first = np.array([[2] * 20, [4] * 20, [4] * 20, [3] * 20])  # every ordering agrees: no noise
    boot = ob.bootstrap_peak(t, first, 7, z=2.0, replicates=25, seed=1)
    assert boot.estimate.maximizing_set.tolist() == [2, 4]
    assert np.all(boot.t_star == boot.estimate.t_star)
    assert boot.interval() == (boot.estimate.t_star, boot.estimate.t_star)
    with_counts = ob.bootstrap_peak(
        t, first, 7, z=2.0, replicates=5, seed=1, exact_counts=np.zeros((4, 1))
    )
    assert np.all(with_counts.t_star == with_counts.estimate.t_star)
    with pytest.raises(ValueError, match="checkpoints, samples"):
        ob.bootstrap_peak(t, first[:3], 7, z=1.0, replicates=5, seed=1)
    gone = ob.bootstrap_peak(t, np.full((4, 5), 8), 7, z=1.0, replicates=5, seed=1)  # k = 0
    assert gone.missing_fraction == 1.0 and gone.estimate.missing
    assert all(math.isnan(v) for v in gone.interval())
