"""The notation tables of study/06 and study/07 list every symbol the documents use, once each.

Review findings T56-F17 and T56-F17-R1: the documentation standard asks each document for a
notation table with "every symbol, its meaning, its type and the code name it has", and the two
documents had symbols with two meanings and symbols used in proofs with no row. These tests read
both documents and check:

* every symbol in the documents' mathematics (prose, displays and tables, outside code) has a row
  in the notation table. A symbol here is a letter (Latin or Greek, or a multi-letter
  ``\\mathrm{...}`` name) with its decorations (hat, bar, tilde, bold, calligraphic,
  blackboard), its primes and a star superscript; subscripts and other superscripts do not change
  it. A subscript or superscript that is only letters and digits is a label or an index (the
  ``c`` of ``p_c``, the ``gc`` of ``R_gc``) and is not read; one with any other content (``m'-1``,
  ``1/\\nu``, ``q,\\nu``) is read for the symbols it uses. A LaTeX command that the reader does
  not know is reported, so a new command must be classified here before it passes;
* each symbol, as it is written in the symbol column (spaces aside), has one row only;
* the letters the review found with two meanings, renamed in the documents, are not used again
  with their old meaning (the patterns of ``RENAMED``, each with what the symbol is now).

The checks are lexical: they cannot see that one listed symbol is used in two senses. That is
what ``RENAMED`` guards for the cases found so far.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("06-statistics.md", "07-finite-size-scaling.md")

GREEK = frozenset(
    {
        *("alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta"),
        *("vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "varpi", "rho"),
        *("varrho", "sigma", "varsigma", "tau", "upsilon", "phi", "varphi", "chi", "psi", "omega"),
        *("Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi"),
        *("Omega", "ell"),
    }
)
# Commands that decorate the symbol they apply to (one argument); synonyms share a key.
DECORATIONS = {
    "hat": r"\hat",
    "widehat": r"\hat",
    "bar": r"\bar",
    "overline": r"\bar",
    "tilde": r"\tilde",
    "widetilde": r"\tilde",
    "mathbf": r"\mathbf",
    "boldsymbol": r"\mathbf",
    "mathcal": r"\mathcal",
    "mathbb": r"\mathbb",
    "mathsf": r"\mathsf",
}
# Commands whose argument is text or an operator name, not symbols.
TEXT_COMMANDS = frozenset({"text", "texttt", "operatorname"})
# Operators, relations, delimiters and spacing: no symbol of their own.
OPERATORS = frozenset(
    {
        *("exp", "ln", "log", "sin", "cos", "tan", "arctan", "min", "max", "sup", "inf", "lim"),
        *("sum", "prod", "int", "sqrt", "frac", "tfrac", "dfrac", "arg", "Pr", "partial"),
        *("big", "Big", "bigg", "Bigg", "left", "right", "lvert", "rvert", "vert", "lbrace"),
        *("rbrace", "lceil", "rceil", "lfloor", "rfloor", "le", "ge", "ne", "leq", "geq", "neq"),
        *("in", "notin", "to", "mapsto", "simeq", "approx", "sim", "propto", "equiv", "gg", "ll"),
        *("subset", "subseteq", "cup", "cap", "setminus", "emptyset", "Rightarrow", "iff", "mid"),
        *("cdot", "cdots", "dots", "ldots", "times", "mp", "pm", "infty", "bmod", "ast"),
        *("quad", "qquad"),
    }
)


def _strip_code(text: str) -> str:
    """The text without fenced code blocks and inline code spans."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    return re.sub(r"`[^`\n]*`", "", text)


def _notation_section(text: str) -> str:
    section = re.search(r"^## 2\. Notation$(.*?)^## ", text, flags=re.S | re.M)
    assert section is not None, "no section '## 2. Notation'"
    return section.group(1)


def _group(math: str, i: int) -> tuple[str, int]:
    """The argument starting at ``math[i]``: a braced group or one token, and the index after."""
    while i < len(math) and math[i] == " ":
        i += 1
    if i >= len(math):
        return "", i
    if math[i] == "{":
        depth = 0
        for j in range(i, len(math)):
            depth += {"{": 1, "}": -1}.get(math[j], 0)
            if depth == 0:
                return math[i + 1 : j], j + 1
        return math[i + 1 :], len(math)
    if math[i] == "\\":
        command = re.match(r"\\([A-Za-z]+|.)", math[i:])
        assert command is not None
        return command.group(0), i + len(command.group(0))
    return math[i], i + 1


def _is_label(script: str) -> bool:
    """A subscript or superscript of letters and digits only: a label or an index, not read."""
    return re.fullmatch(r"[A-Za-z0-9]*", re.sub(r"\\[,;! ]|[\s{}]", "", script)) is not None


def symbols(
    math: str, unknown: set[str], *, scripts: bool = True, in_script: bool = False
) -> list[str]:
    """The symbols (keys) of one math span, in order.

    With ``scripts=False`` the contents of subscripts and superscripts are skipped (the symbol
    column of the notation table, where ``t_{q,\\nu}`` lists ``t``).
    """
    out: list[str] = []
    i = 0
    while i < len(math):
        base = None
        if math[i] == "\\":
            command = re.match(r"\\([A-Za-z]+|.)", math[i:])
            assert command is not None
            name = command.group(1)
            i += len(command.group(0))
            if name in DECORATIONS:
                argument, i = _group(math, i)
                inner = symbols(argument, unknown, scripts=scripts, in_script=in_script)
                base = DECORATIONS[name] + " " + (inner[0] if inner else argument.strip())
                out.extend(inner[1:])
            elif name in GREEK:
                base = "\\" + name
            elif name == "mathrm":
                argument, i = _group(math, i)
                if in_script:  # a label such as N_{\mathrm{free}}
                    continue
                base = r"\mathrm{" + argument + "}"
            elif name in TEXT_COMMANDS:
                _, i = _group(math, i)
                continue
            else:
                if name.isalpha() and name not in OPERATORS:
                    unknown.add(name)
                continue
        elif math[i].isalpha():
            base = math[i]
            i += 1
        else:
            i += 1
            continue
        primes, star = "", False
        while i < len(math) and math[i] in "_^'":
            if math[i] == "'":
                primes += "'"
                i += 1
                continue
            kind = math[i]
            argument, i = _group(math, i + 1)
            if kind == "^" and re.match(r"\s*(\\ast\b|\*)", argument):
                star = True
                argument = re.sub(r"^\s*(\\ast\b|\*)", "", argument)
            if scripts and not _is_label(argument):
                out.extend(symbols(argument, unknown, in_script=True))
        out.append(base + primes + ("^*" if star else ""))
    return out


def _spans(text: str) -> list[tuple[str, bool]]:
    """Every math span of ``text`` outside code, with whether it is in a table's first column."""
    spans: list[tuple[str, bool]] = []
    for piece in re.split(r"(\$\$.*?\$\$)", _strip_code(text), flags=re.S):
        if piece.startswith("$$"):
            spans.append((piece[2:-2], False))
            continue
        for line in piece.splitlines():
            for match in re.finditer(r"\$([^$\n]+)\$", line):
                first_column = line.startswith("|") and line[: match.start()].count("|") == 1
                spans.append((match.group(1), first_column))
    return spans


# Landau's O written as a plain O (not \mathcal O).
_LANDAU = r"(?<!\\mathcal )(?<!\\mathcal\{)(?<![\\A-Za-z])O\("

# Symbols renamed after the review found them with two meanings (T56-F17, T56-F17-R1): a pattern
# of the old use, searched in the math spans outside the notation table, and what it is now.
RENAMED: dict[str, tuple[tuple[str, str], ...]] = {
    "06-statistics.md": (
        (
            r"\b[mnN]\(G\)|G = g|\\mathbb\{E\}_\{?G",
            "a random graph is \\mathcal G; G counts graphs",
        ),
        (
            r"\\mu_\{n",
            "the Watts-Strogatz law is WS(n, beta) (v1.4's mu_{n,beta}, named in the table)",
        ),
        (r"\\hat m\b", "the nested mean of ln T* is \\hat\\eta"),
        (
            r"A\(t\)|\\lvert T\\rvert",
            "Student's variable is X_nu, its value s, A a function of theta",
        ),
        (r"\bn_g\b|\bn_\{g\}", "the rows of graph g are N_g (n is the system size)"),
        (r"\b[ab]_k\b|\b[ab]_\{k", "the Student sums' coefficients are alpha_k and kappa_k"),
        (r"\\ast ?b\b", "a bootstrap replicate is starred, without the index b"),
        (r"\bx \\bmod", "a raw 64-bit word is omega"),
        (_LANDAU, "an order of growth is \\mathcal O"),
        (r"\be\^", "the exponential is \\exp (e is a leaf effect)"),
    ),
    "07-finite-size-scaling.md": (
        (r"P_A", "the projection is \\Pi_A (P counts points)"),
        (
            r"\\ell_k|\bh_k",
            "the box of coordinate q is [vartheta_q^-, vartheta_q^+] (ell is a threshold)",
        ),
        (
            r"\\theta_k",
            "a coordinate of the parameters is vartheta_q (theta is the time observable)",
        ),
        (
            r"\b[pysxY]'(?![_'])|S'_o|\\nu'|dY'",
            "mapped data are written with a tilde (x' is another size's)",
        ),
        (r"\bg_i\b", "the objective's derivative in x_i is gamma_i"),
        (r"_\{m-[12]\}", "the numbers of points of n' and n'' are m' and m''"),
        (_LANDAU, "an order of growth is \\mathcal O (O counts observables)"),
        (r"\\mathrm\{se\}", "the methods' errors are named by their fields"),
        (r"\be\^", "the exponential is \\exp (e_l is a point's error)"),
    ),
}


def notation_problems(doc: str, text: str) -> list[str]:
    """Symbols used without a row, symbols with two rows, unknown commands, renamed symbols."""
    problems: list[str] = []
    unknown: set[str] = set()
    table = _notation_section(text)
    listed: dict[str, int] = {}
    rows: dict[str, int] = {}
    number = 0
    for line in table.splitlines():
        if not line.startswith("| $"):
            continue
        number += 1
        for symbol in re.findall(r"\$([^$]+)\$", line.split("|")[1]):
            written = re.sub(r"\s+", "", symbol)
            if written in rows and rows[written] != number:
                problems.append(f"study/{doc}: ${symbol}$ has two rows in the notation table")
            rows[written] = number
            for key in symbols(symbol, unknown, scripts=False):
                listed.setdefault(key, number)
    assert listed, f"study/{doc}: the notation table has no rows"
    missing: dict[str, str] = {}
    outside = [span for span, _ in _spans(text.replace(table, ""))]
    # the notation table's other columns use symbols too; its first column is the rows themselves
    meanings = [span for span, first_column in _spans(table) if not first_column]
    for span in outside + meanings:
        for key in symbols(span, unknown):
            if key not in listed:
                missing.setdefault(key, span.strip())
    for key, span in missing.items():
        problems.append(f"study/{doc}: ${key}$ (in ${span[:60]}$) has no row in the notation table")
    for name in sorted(unknown):
        problems.append(f"study/{doc}: the command \\{name} is not classified in this test")
    for pattern, now in RENAMED[doc]:
        for span in outside:
            if re.search(pattern, span):
                problems.append(f"study/{doc}: ${span.strip()[:60]}$ uses an old symbol: {now}")
    return problems


def _text(doc: str) -> str:
    return (ROOT / "docs" / "study" / doc).read_text(encoding="utf-8")


@pytest.mark.parametrize("doc", DOCS)
def test_the_notation_table_lists_every_symbol_once(doc):
    assert notation_problems(doc, _text(doc)) == []


def _replace_last(text: str, old: str, new: str) -> str:
    """``text`` with the last occurrence of ``old`` (after the notation table) replaced."""
    at = text.rfind(old)
    assert at > text.find("## 3."), old
    return text[:at] + new + text[at + len(old) :]


def test_the_checks_find_a_missing_row_a_doubled_row_and_an_old_symbol():
    doc = "06-statistics.md"
    text = _text(doc)

    def planted(old: str, new: str) -> list[str]:
        assert text.count(old) >= 1, old
        return notation_problems(doc, text.replace(old, new, 1))

    # a row removed: K (values per row) is used in the costs of section 9
    row = next(line for line in text.splitlines() if line.startswith("| $K$ |"))
    problems = planted(row + "\n", "")
    assert any(p.startswith("study/06-statistics.md: $K$ (in") for p in problems), problems
    # a symbol used in a proof that no row lists
    problems = planted("$\\hat\\sigma^2 = ", "$\\hat\\sigma^2 = \\varkappa + ")
    assert any("\\varkappa is not classified" in p for p in problems), problems
    problems = planted("$\\hat\\sigma^2 = ", "$\\hat\\sigma^2 = Q' + ")
    assert any(p.startswith("study/06-statistics.md: $Q'$ (in") for p in problems), problems
    # a symbol in two rows
    problems = planted("| $L$ |", "| $L$, $B$ |")
    assert "study/06-statistics.md: $B$ has two rows in the notation table" in problems
    # a renamed symbol used with its old meaning
    problems = notation_problems(doc, _replace_last(text, "\\hat\\eta", "\\hat m"))
    assert any("uses an old symbol: the nested mean of ln T*" in p for p in problems), problems

    doc = "07-finite-size-scaling.md"
    text = _text(doc)
    assert notation_problems(doc, text) == []
    mapped = _replace_last(text, "\\tilde x", "x'")
    assert any("uses an old symbol: mapped data" in p for p in notation_problems(doc, mapped))
    # a script that is an expression is read: m'' must have its row
    row = next(line for line in text.splitlines() if line.startswith("| $n'$") and "$m''$" in line)
    cut = text.replace(row, row.replace("$m''$", "$m'''$"), 1)
    problems = notation_problems(doc, cut)
    assert any(p.startswith("study/07-finite-size-scaling.md: $m''$ (in") for p in problems)
