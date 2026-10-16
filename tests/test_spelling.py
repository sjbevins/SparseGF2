"""American spelling everywhere: code, comments, docstrings, messages, names and documents.

The word list is Clank's (``tools/check_spelling.py`` in the Clank repository), with the forms this repository had
used added. A British form may stay only where changing it would be wrong:

* a line marked ``spelling: keep``, which says why (a quotation from a paper, the title of a paper);
* a value in ``PUBLIC``, written as a literal (in quotes or backticks, or in a ``|`` list of values): it is part of
  the public interface, so renaming it would break the clients that compare against it, and it waits for the owner;
* Markdown code blocks and inline code, which quote code, values and screens whose own files are checked here (or
  that live on another branch, as in the agent-interface plan).

This file holds the list, so it is not scanned.
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SELF = "tests/test_spelling.py"
KEEP = "spelling: keep"
BINARY = {".png", ".npz", ".npy", ".pdf", ".ico", ".jpg", ".jpeg", ".gif", ".pyc", ".zip"}

AMERICAN = {
    # Clank's list
    "aeroplane": "airplane",
    "ageing": "aging",
    "amongst": "among",
    "analogue": "analog",
    "analyse": "analyze",
    "analysed": "analyzed",
    "analysing": "analyzing",
    "apologise": "apologize",
    "artefact": "artifact",
    "artefacts": "artifacts",
    "authorisation": "authorization",
    "authorise": "authorize",
    "authorised": "authorized",
    "behaviour": "behavior",
    "behavioural": "behavioral",
    "behaviours": "behaviors",
    "cancelled": "canceled",
    "cancelling": "canceling",
    "catalogue": "catalog",
    "catalogued": "cataloged",
    "catalogues": "catalogs",
    "categorise": "categorize",
    "categorised": "categorized",
    "centre": "center",
    "centred": "centered",
    "centres": "centers",
    "cheque": "check",
    "colour": "color",
    "coloured": "colored",
    "colourful": "colorful",
    "colouring": "coloring",
    "colours": "colors",
    "customise": "customize",
    "customised": "customized",
    "defence": "defense",
    "emphasise": "emphasize",
    "enrolment": "enrollment",
    "favour": "favor",
    "favoured": "favored",
    "favourite": "favorite",
    "favourites": "favorites",
    "favours": "favors",
    "fibre": "fiber",
    "finalise": "finalize",
    "finalised": "finalized",
    "fulfil": "fulfill",
    "grey": "gray",
    "honour": "honor",
    "initialise": "initialize",
    "initialised": "initialized",
    "judgement": "judgment",
    "labelled": "labeled",
    "labelling": "labeling",
    "labour": "labor",
    "learnt": "learned",
    "licence": "license",
    "licences": "licenses",
    "litre": "liter",
    "manoeuvre": "maneuver",
    "maximise": "maximize",
    "maximised": "maximized",
    "metre": "meter",
    "metres": "meters",
    "minimise": "minimize",
    "minimised": "minimized",
    "modelled": "modeled",
    "modelling": "modeling",
    "mould": "mold",
    "neighbour": "neighbor",
    "neighbouring": "neighboring",
    "neighbours": "neighbors",
    "normalise": "normalize",
    "normalised": "normalized",
    "offence": "offense",
    "optimisation": "optimization",
    "optimise": "optimize",
    "optimised": "optimized",
    "optimising": "optimizing",
    "organisation": "organization",
    "organisations": "organizations",
    "organise": "organize",
    "organised": "organized",
    "organises": "organizes",
    "organising": "organizing",
    "practise": "practice",
    "prioritise": "prioritize",
    "prioritised": "prioritized",
    "programme": "program",
    "programmes": "programs",
    "realise": "realize",
    "realised": "realized",
    "realises": "realizes",
    "recognise": "recognize",
    "recognised": "recognized",
    "recognises": "recognizes",
    "rumour": "rumor",
    "sceptical": "skeptical",
    "serialise": "serialize",
    "serialised": "serialized",
    "signalled": "signaled",
    "skilful": "skillful",
    "standardise": "standardize",
    "summarise": "summarize",
    "summarised": "summarized",
    "synchronise": "synchronize",
    "synchronised": "synchronized",
    "theatre": "theater",
    "towards": "toward",
    "travelled": "traveled",
    "travelling": "traveling",
    "tyre": "tire",
    "utilise": "utilize",
    "visualise": "visualize",
    "visualised": "visualized",
    "whilst": "while",
    # forms this repository had used
    "amortise": "amortize",
    "amortised": "amortized",
    "centring": "centering",
    "characterise": "characterize",
    "characterised": "characterized",
    "characterises": "characterizes",
    "colourmap": "colormap",
    "favouring": "favoring",
    "generalise": "generalize",
    "generalised": "generalized",
    "generalises": "generalizes",
    "initialisation": "initialization",
    "initialises": "initializes",
    "journalling": "journaling",
    "materialise": "materialize",
    "materialised": "materialized",
    "materialises": "materializes",
    "mislabelled": "mislabeled",
    "neighbourhood": "neighborhood",
    "neighbourhoods": "neighborhoods",
    "normalises": "normalizes",
    "normalising": "normalizing",
    "regularise": "regularize",
    "regularised": "regularized",
    "relabelled": "relabeled",
    "serialising": "serializing",
    "stabiliser": "stabilizer",
    "stabilisers": "stabilizers",
    "summarises": "summarizes",
    "summarising": "summarizing",
}

PUBLIC = {
    # the status of a stopped background job: api.job_status and list_jobs, the MCP job_status and
    # list_jobs tools and the CLI report it, and clients compare against it
    "cancelled": "the status value of a background job",
}

# A British form standing alone: letters on neither side, so `_`, digits and punctuation mark its edges and the
# forms inside names (`_is_normalised`) are found too.
WORD = re.compile(
    r"(?<![A-Za-z])(" + "|".join(sorted(AMERICAN, key=len, reverse=True)) + r")(?![A-Za-z])",
    re.IGNORECASE,
)
LITERAL_EDGE = set("\"'`|")
FENCE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE = re.compile(r"(`+)(?:(?!\1).)+?\1")


def _prose_lines(text: str) -> list[tuple[int, str]]:
    """The lines of a Markdown text outside code blocks, with inline code blanked out."""
    lines, fenced = [], False
    for number, line in enumerate(text.splitlines(), start=1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            lines.append((number, INLINE_CODE.sub(lambda m: " " * len(m.group(0)), line)))
    return lines


def _public_literal(line: str, match: re.Match) -> bool:
    """A value of ``PUBLIC`` written as a literal: quoted, in backticks or in a ``|`` list."""
    if match.group(1) not in PUBLIC:
        return False
    before = line[match.start() - 1] if match.start() > 0 else ""
    after = line[match.end()] if match.end() < len(line) else ""
    return before in LITERAL_EDGE and after in LITERAL_EDGE


def scan_text(path: str, text: str) -> list[str]:
    """The British spellings of one file, as ``path:line: word -> american``."""
    lines = (
        _prose_lines(text) if path.endswith(".md") else list(enumerate(text.splitlines(), start=1))
    )
    return [
        f"{path}:{number}: {match.group(1)} -> {AMERICAN[match.group(1).lower()]}"
        for number, line in lines
        if KEEP not in line
        for match in WORD.finditer(line)
        if not _public_literal(line, match)
    ]


def _repository_files() -> list[str]:
    try:
        listed = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=True
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout: the tracked files are unknown")
    return [name for name in listed if name != SELF and Path(name).suffix.lower() not in BINARY]


def test_the_repository_is_spelled_in_american_english():
    findings = []
    for name in _repository_files():
        try:
            text = (ROOT / name).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        findings.extend(scan_text(name, text))
    assert not findings, "British spellings (the American form after ->):\n" + "\n".join(findings)


def test_the_scanner_finds_british_forms_and_keeps_only_what_it_should():
    code = 'x = "normalised"  # colour\ny = _is_normalised(a)  # neighbours\nok = "cancelled"\n'
    assert [f.split(": ", 1)[1] for f in scan_text("a.py", code)] == [
        "normalised -> normalized",
        "colour -> color",
        "normalised -> normalized",
        "neighbours -> neighbors",
    ]
    # the public value is kept only as a literal; in prose it is a spelling like any other
    assert scan_text("a.py", "status (queued|running|cancelled|failed)\n") == []
    assert scan_text("a.py", "# a cancelled job\n") == ["a.py:1: cancelled -> canceled"]
    text = "The Colour\n```python\ncolour = 1\n```\nuse `colour_of` here, a Behaviour\nquoted (spelling: keep) colour\n"
    assert scan_text("a.md", text) == ["a.md:1: Colour -> color", "a.md:5: Behaviour -> behavior"]
