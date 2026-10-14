"""Tests of the test harness itself: warnings are errors, and skips cannot hide."""

import warnings
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

CONFTEST = Path(__file__).with_name("conftest.py")


def test_warnings_are_errors():
    with pytest.raises(UserWarning):
        warnings.warn("x", UserWarning, stacklevel=2)


def _inner(pytester, monkeypatch, test_body, no_skips):
    # The inner session runs the real conftest, so the real gate is what is tested.
    pytester.makeconftest(CONFTEST.read_text(encoding="utf-8"))
    pytester.makepyfile(test_body)
    if no_skips:
        monkeypatch.setenv("SPARSEGF2_NO_SKIPS", "1")
    else:
        monkeypatch.delenv("SPARSEGF2_NO_SKIPS", raising=False)
    # No doctest plugin: the inner session only needs to collect the one test file.
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", "-p", "no:doctest")


SKIPPING = """
import pytest

def test_a():
    pytest.skip("x")
"""

ALLOWED = """
import pytest

def test_a():
    pytest.skip("no working bash")
"""


def test_no_skip_gate_fails_a_skipping_session(pytester, monkeypatch):
    result = _inner(pytester, monkeypatch, SKIPPING, no_skips=True)
    result.assert_outcomes(skipped=1)
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*test_a*x*"])


def test_skips_pass_without_the_gate(pytester, monkeypatch):
    result = _inner(pytester, monkeypatch, SKIPPING, no_skips=False)
    result.assert_outcomes(skipped=1)
    assert result.ret == 0


def test_no_skip_gate_allows_the_no_bash_reason(pytester, monkeypatch):
    result = _inner(pytester, monkeypatch, ALLOWED, no_skips=True)
    result.assert_outcomes(skipped=1)
    assert result.ret == 0


MIXED = """
import pytest

def test_allowed():
    pytest.skip("no working bash")

def test_other():
    pytest.skip("x")
"""


def test_no_skip_gate_names_only_the_disallowed_skip(pytester, monkeypatch):
    result = _inner(pytester, monkeypatch, MIXED, no_skips=True)
    result.assert_outcomes(skipped=2)
    assert result.ret != 0
    out = result.stdout.str()
    assert "test_other" in out.split("SPARSEGF2_NO_SKIPS=1")[-1]
    assert "test_allowed" not in out.split("SPARSEGF2_NO_SKIPS=1")[-1]
