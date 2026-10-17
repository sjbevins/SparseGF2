"""What the T2 documents say about the suite and their notation (review finding T2-F21).

Two documents quote the test suite and define notation for it,
``docs/study/01-ensemble-and-sampling.md`` and ``docs/study/02-identity-seeds-and-manifest.md``.
Such quotes go stale silently. Study/02's validation map once said 130 malformed-manifest cases
when the suite had 131. Study/01 used the letter s for the symplectic class while study/02, which
study/01 cites for the stride, used s for the circuit index stride. Study/02 also kept G_* after
v1.4 renamed it N_graphs. The tests below read the documents and collect the T2 test files with
``pytest --collect-only``, which runs none of the tests. They check:

* every test a document names is collected, from the file it gives (``tests/f.py::test_x``, and
  the names after it in the same table cell) or else from one of the three T2 test files, and a
  name ending in ``*`` matches at least one test;
* a count written right after a parametrized test's name (`` `test_x` (N ...``) is that test's
  number of collected cases (``parametrize`` marks and parametrized fixtures alike), a count of a
  wildcard ("(N tests") is the number of tests it matches, and the counts the validation map
  writes in words (``WORD_CLAIMS``) are the suite's;
* a notation table lists each symbol in one row, the symbols both tables list are among the five
  that mean the same thing in both (n, beta, p, g, u), and G_* appears only in the note that
  earlier drafts of v1.4 wrote it.

The last test plants each of these errors in the current text and requires that the checks find
it, so they cannot pass simply because they no longer look at anything.
"""

from __future__ import annotations

import ast
import functools
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_01 = ROOT / "docs" / "study" / "01-ensemble-and-sampling.md"
DOC_02 = ROOT / "docs" / "study" / "02-identity-seeds-and-manifest.md"
T2_FILES = ("test_study_identity.py", "test_study_manifest.py", "test_study_sampling.py")

# A test named in backticks, with its file when given, and a count written right after it.
_CITED = re.compile(
    r"`(?:tests/(?P<file>\w+\.py)::)?(?P<name>test_\w+)(?P<star>\*)?`"
    r"(?: \((?P<count>\d+) (?P<word>\w+))?"
)

# The symbols both notation tables list. Each was checked by hand to mean the same thing in both
# documents; a new shared symbol must be checked the same way and added, or renamed in one table.
SHARED_SYMBOLS = frozenset({"n", r"\beta", "p", "g", "u"})

# v1.4 (commit fb9062e) renamed the number of graphs per condition G_* to N_graphs.
_OLD_GRAPH_COUNT = re.compile(r"G_\*|G_\{\*\}|G_\{?\\ast")
_ALIAS_NOTE = "earlier drafts of v1.4 wrote"

# Counts the documents write in words: (document, pattern of the count, test file, test, what is
# counted: None for the test's parametrized cases, or the name of a list literal in the test).
WORD_CLAIMS = (
    (
        "02",
        r"among the (\d+) malformed cases",
        "test_study_manifest.py",
        "test_malformed_manifests_are_refused_naming_the_field",
        None,
    ),
    (
        "02",
        r"\| (\d+) bad inputs",
        "test_study_identity.py",
        "test_malformed_tokens_are_refused_with_a_message_naming_the_field",
        None,
    ),
    (
        "02",
        r"exact: (\d+) listed spellings",
        "test_study_identity.py",
        "test_canonical_token_strings",
        None,
    ),
    (
        "02",
        r"\| (\d+) manifests built with",
        "test_study_manifest.py",
        "test_direct_construction_and_replace_obey_the_rules_of_reading",
        "cases",
    ),
)


def _texts() -> dict[str, str]:
    return {
        "01": DOC_01.read_text(encoding="utf-8"),
        "02": DOC_02.read_text(encoding="utf-8"),
    }


@functools.cache
def _tree(file: str) -> ast.Module:
    return ast.parse((ROOT / "tests" / file).read_text(encoding="utf-8"))


@functools.cache
def _collect(files: tuple[str, ...]) -> dict[str, dict[str, tuple[str | None, ...]]]:
    """What pytest collects from ``tests/<file>`` for each file: test name -> parameter ids
    (``(None,)`` for a test without parameters). Collecting runs no test; it counts the cases of
    ``parametrize`` marks and of parametrized fixtures as the suite does."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
        + ["-o", "addopts="]
        + [f"tests/{file}" for file in files],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    ids: dict[str, dict[str, list[str | None]]] = {file: {} for file in files}
    for line in result.stdout.splitlines():
        match = re.fullmatch(r"tests/(\w+\.py)::(test_\w+)(?:\[(.*)\])?", line.strip())
        if match:
            ids[match[1]].setdefault(match[2], []).append(match[3])
    return {file: {name: tuple(v) for name, v in tests.items()} for file, tests in ids.items()}


def _collected(file: str) -> dict[str, tuple[str | None, ...]]:
    return _collect(T2_FILES)[file] if file in T2_FILES else _collect((file,))[file]


def _number_of_cases(file: str, name: str) -> int | None:
    """The number of cases of a parametrized test as pytest collects them, else None."""
    ids = _collected(file)[name]
    return None if ids == (None,) else len(ids)


def _list_length(file: str, name: str, variable: str) -> int:
    """The number of elements of the list literal assigned to ``variable`` in test ``name``."""
    for node in _tree(file).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Assign)
                    and isinstance(sub.value, ast.List)
                    and any(isinstance(t, ast.Name) and t.id == variable for t in sub.targets)
                ):
                    return len(sub.value.elts)
    raise AssertionError(f"{file}::{name} assigns no list literal to {variable}")


def _citations(text: str) -> Iterator[tuple[re.Match[str], str | None]]:
    """Each test name in backticks with the files it is looked up in and the count after it.

    The documents write ``tests/f.py::test_a``, ``test_b`` for two tests of f: within one table
    cell (or one line of prose), a name without a file belongs to the file named last before it,
    and otherwise to one of the T2 test files.
    """
    for line in text.splitlines():
        for cell in line.split("|"):
            current = None
            for match in _CITED.finditer(cell):
                current = match["file"] or current
                yield match, match["file"] or current


def cited_test_problems(texts: dict[str, str]) -> list[str]:
    """Named tests that do not exist, and counts after a test's name that are not the suite's."""
    problems = []
    for doc, text in texts.items():
        for match, file in _citations(text):
            name, star, count, word = match.group("name", "star", "count", "word")
            if file and not (ROOT / "tests" / file).is_file():
                problems.append(f"study/{doc}: tests/{file} does not exist")
                continue
            files = (file,) if file else T2_FILES
            if star:
                matched = [t for f in files for t in _collected(f) if t.startswith(name)]
                if not matched:
                    problems.append(f"study/{doc}: no test in {', '.join(files)} matches {name}*")
                elif count is not None and word == "tests" and int(count) != len(matched):
                    problems.append(
                        f"study/{doc}: {name}* ({count} tests) but {len(matched)} tests match"
                    )
                continue
            homes = [f for f in files if name in _collected(f)]
            if not homes:
                problems.append(f"study/{doc}: {name} is not defined in {', '.join(files)}")
                continue
            if count is not None:
                cases = _number_of_cases(homes[0], name)
                if cases is not None and cases != int(count):
                    problems.append(
                        f"study/{doc}: {name} ({count} {word} ...) but the suite has {cases} cases"
                    )
    return problems


def word_claim_problems(texts: dict[str, str]) -> list[str]:
    """Counts written in words (``WORD_CLAIMS``) that are not the suite's, or no longer written."""
    problems = []
    for doc, pattern, file, name, variable in WORD_CLAIMS:
        found = re.findall(pattern, texts[doc])
        if not found:
            problems.append(f"study/{doc}: the claim {pattern!r} is no longer written")
            continue
        if variable is None:
            actual = _number_of_cases(file, name)
            assert actual is not None, f"{name} is not parametrized"
        else:
            actual = _list_length(file, name, variable)
        for count in found:
            if int(count) != actual:
                problems.append(f"study/{doc}: {pattern!r} reads {count}, the suite has {actual}")
    return problems


def _notation_rows(text: str) -> list[list[str]]:
    """The symbols of each row of the notation table (section 2), one list per row."""
    section = re.search(r"^## 2\. Notation$(.*?)^## ", text, flags=re.S | re.M)
    assert section is not None, "no section '## 2. Notation'"
    rows = []
    for line in section.group(1).splitlines():
        if line.startswith("|"):
            symbols = re.findall(r"\$([^$]+)\$", line.split("|")[1])
            if symbols:
                rows.append(symbols)
    assert rows, "the notation table has no rows"
    return rows


def notation_problems(texts: dict[str, str]) -> list[str]:
    """A symbol in two rows of one table, a symbol shared between the tables that is not known to
    mean the same thing in both, and G_* outside the note on v1.4's earlier drafts."""
    problems = []
    tables: dict[str, set[str]] = {}
    for doc, text in texts.items():
        listed: set[str] = set()
        for symbols in _notation_rows(text):
            for symbol in symbols:
                if symbol in listed:
                    problems.append(f"study/{doc}: ${symbol}$ has two rows in the notation table")
                listed.add(symbol)
        tables[doc] = listed
        for number, line in enumerate(text.splitlines(), start=1):
            if _OLD_GRAPH_COUNT.search(line) and _ALIAS_NOTE not in line:
                problems.append(
                    f"study/{doc}, line {number}: G_* is v1.4's earlier name; v1.4 writes N_graphs"
                )
    shared = set.intersection(*tables.values())
    for symbol in sorted(shared - SHARED_SYMBOLS):
        problems.append(
            f"${symbol}$ is in the notation tables of both documents: rename it in one, or check"
            " that it means the same thing in both and add it to SHARED_SYMBOLS"
        )
    return problems


def test_the_tests_the_t2_documents_name_exist_and_the_counts_after_them_are_the_suites():
    problems = cited_test_problems(_texts())
    assert not problems, "\n".join(problems)


def test_the_counts_the_t2_documents_write_in_words_are_the_suites():
    problems = word_claim_problems(_texts())
    assert not problems, "\n".join(problems)


def test_the_t2_notation_tables_give_each_symbol_one_meaning_and_use_the_symbols_of_v14():
    problems = notation_problems(_texts())
    assert not problems, "\n".join(problems)


def test_the_checks_find_the_errors_of_finding_t2_f21():
    texts = _texts()
    malformed = "test_malformed_manifests_are_refused_naming_the_field"
    cases = _number_of_cases("test_study_manifest.py", malformed)
    assert cases is not None

    def planted(doc: str, old: str, new: str) -> dict[str, str]:
        assert texts[doc].count(old) >= 1, f"study/{doc} no longer contains {old!r}"
        return {**texts, doc: texts[doc].replace(old, new, 1)}

    # a stale count after a parametrized test's name, as the 130 of 131
    stale = cited_test_problems(planted("02", f"({cases} cases)", f"({cases - 1} cases)"))
    assert len(stale) == 1 and malformed in stale[0] and f"has {cases} cases" in stale[0]
    # a test that no longer exists, and a wildcard that matches nothing
    gone = cited_test_problems(planted("02", "`test_allowed_revisions`", "`test_allowed_revision`"))
    assert gone == ["study/02: test_allowed_revision is not defined in " + ", ".join(T2_FILES)]
    star = cited_test_problems(planted("02", "`test_trajectory_key_*`", "`test_trajectory_keys_*`"))
    assert len(star) == 1 and "matches test_trajectory_keys_*" in star[0]
    # a wrong number of tests behind a wildcard (study/02 says five)
    tests = cited_test_problems(planted("02", "key_*` (5 tests", "key_*` (6 tests"))
    assert len(tests) == 1 and "(6 tests) but 5 tests match" in tests[0]
    # a count of a test parametrized by a fixture (the two rewiring probabilities of study/01)
    fixture = cited_test_problems(planted("01", "deletion` (2 values", "deletion` (3 values"))
    assert len(fixture) == 1 and "the suite has 2 cases" in fixture[0]
    # a stale count in words, and a claim whose wording changed
    words = word_claim_problems(
        planted("02", f"among the {cases} malformed", f"among the {cases + 1} malformed")
    )
    assert len(words) == 1 and f"reads {cases + 1}, the suite has {cases}" in words[0]
    reworded = word_claim_problems(planted("02", "manifests built with", "manifests made with"))
    assert len(reworded) == 1 and "no longer written" in reworded[0]
    # the letter clash: study/02 listing s (study/01's symplectic class) for the stride
    clash = notation_problems(planted("02", r"| $\bar C$ |", "| $s$ |"))
    assert len(clash) == 1 and clash[0].startswith("$s$ is in the notation tables of both")
    # one symbol in two rows of one table
    twice = notation_problems(planted("01", r"| $\beta$ |", "| $p$ |"))
    assert "study/01: $p$ has two rows in the notation table" in twice
    # v1.4's earlier symbol outside the note (the row of u, which has no note)
    index = r"the trajectory index $u = g \bar C + c$"
    old = notation_problems(planted("02", index, index + r" for $g < G_*$"))
    assert len(old) == 1 and "G_* is v1.4's earlier name" in old[0]
    # and the note itself is allowed
    assert notation_problems(texts) == []
