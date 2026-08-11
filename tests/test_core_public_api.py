"""Public API smoke test.

Ensures every symbol pinned in :mod:`sparsegf2.__init__` is importable
and roundtrips through a small workload. This is the "stable surface"
the circuits package builds on.

Failing this test means the public API has drifted; update the expected
set below (and the changelog) when that is intentional.
"""

from __future__ import annotations

import numpy as np
import pytest

import sparsegf2


def test_version_is_set():
    """sparsegf2.__version__ must be a string (PEP 440-ish)."""
    assert isinstance(sparsegf2.__version__, str)
    assert sparsegf2.__version__


def test_all_exports_resolvable():
    """Every name in __all__ must be a real attribute of the package."""
    for name in sparsegf2.__all__:
        assert hasattr(sparsegf2, name), (
            f"sparsegf2.__all__ lists {name!r} but it's not an attribute"
        )


# Set of symbols the circuits package will rely on. Pinning here forces
# anyone removing a symbol to also update this list (and presumably the
# circuits package).
_EXPECTED_PUBLIC = {
    # Core class
    "SparseGF2",
    # Factories
    "from_bell_purification",
    "from_zero_state",
    # Named symplectic constants
    "CX_SYMP",
    "CZ_SYMP",
    "H_SYMP",
    "PAULI_I",
    "PAULI_X",
    "PAULI_Y",
    "PAULI_Z",
    "S_SYMP",
    "SQRT_X_SYMP",
    "SWAP_SYMP",
    # Sp(2n, F_2) machinery
    "enumerate_sp4",
    "enumerate_symplectic_group",
    "is_symplectic",
    "random_symplectic",
    "random_symplectic_2q",
    "random_symplectic_2q_batch",
    "random_symplectic_batch",
    "symplectic_form",
    "symplectic_group_order",
    "symplectic_product",
    # Observables
    "Subsystem",
    "average_stabilizer_weight",
    "code_dimension",
    "code_rate",
    "complement",
    "contiguous_distance",
    "entanglement_entropy",
    "generator_weights",
    "mutual_information",
    "single_qubit_entropy",
    "stabilizer_weight_spectrum",
    "subsystem_rank",
    "tripartite_mutual_info",
    # Exception hierarchy (for downstream `except SparseGF2Error:` catch-all)
    "InvalidArgumentError",
    "SimulatorBackendError",
    "SparseGF2Error",
    "SymplecticConditionError",
    "TableauCorruption",
    # Package metadata
    "__version__",
}


def test_public_api_matches_expected_set():
    """Snapshot test: __all__ contains exactly the expected symbols.

    Add/remove from `_EXPECTED_PUBLIC` and `__all__` in lockstep, and
    note the change in the changelog when you do.
    """
    actual = set(sparsegf2.__all__)
    missing = _EXPECTED_PUBLIC - actual
    extra = actual - _EXPECTED_PUBLIC
    assert not missing, f"sparsegf2.__all__ is missing: {sorted(missing)}"
    assert not extra, f"sparsegf2.__all__ has unexpected entries: {sorted(extra)}"


def test_end_to_end_roundtrip():
    """One-shot smoke through the whole public surface."""
    from sparsegf2 import (
        CX_SYMP,
        H_SYMP,
        SparseGF2,
        code_dimension,
        entanglement_entropy,
        from_bell_purification,
        is_symplectic,
        random_symplectic,
        random_symplectic_2q,
        stabilizer_weight_spectrum,
        symplectic_group_order,
    )

    # 1. Build a Bell-purification state, check observables.
    sim = from_bell_purification(3)
    assert code_dimension(sim, n_system=3) == 3
    assert entanglement_entropy(sim, [0, 1, 2]) == 3
    spec = stabilizer_weight_spectrum(sim)
    assert spec.sum() == sim.n  # one stabilizer per qubit

    # 2. Apply some gates.
    sim.apply_h(0)
    sim.apply_gate_2q(0, 1, CX_SYMP)
    sim.apply_gate_1q(2, H_SYMP)

    # 3. Sample a random symplectic at n=3, check it's symplectic, apply it.
    rng = np.random.default_rng(0)
    S = random_symplectic(3, rng=rng)
    assert is_symplectic(S)

    S2 = random_symplectic_2q(rng=rng)
    assert is_symplectic(S2)
    sim_small = SparseGF2(2)
    sim_small.apply_gate_2q(0, 1, S2)

    # 4. Group-order sanity.
    assert symplectic_group_order(3) == 1_451_520


def test_observables_module_is_importable():
    """`from sparsegf2.core.observables import *` should not error."""
    import sparsegf2.core.observables as obs

    expected_attrs = (
        "subsystem_rank",
        "entanglement_entropy",
        "mutual_information",
        "single_qubit_entropy",
        "tripartite_mutual_info",
        "code_dimension",
        "code_rate",
        "contiguous_distance",
        "generator_weights",
        "stabilizer_weight_spectrum",
        "average_stabilizer_weight",
    )
    for a in expected_attrs:
        assert hasattr(obs, a), f"observables module missing {a}"


def test_no_runtime_stim_import():
    """The core simulator must never import ``stim`` at runtime.

    Three layers of defense:

    1. **Source grep**: no `import stim` / `from stim` literal under
       ``src/sparsegf2/``. Catches both module-level and function-local
       (lazy) imports, the latter of which would be invisible to the
       ``__dict__`` check.
    2. **``__dict__``**: no imported ``sparsegf2.*`` module has ``stim``
       as a global attribute. Catches module-level imports.
    3. **``sys.modules``**: optional belt-and-braces, but note that the
       tests *themselves* import ``stim``, so we don't assert
       ``"stim" not in sys.modules`` here.
    """
    import pathlib
    import re
    import sys

    src_root = pathlib.Path(__file__).resolve().parent.parent / "src" / "sparsegf2"
    pattern = re.compile(r"^\s*(?:import\s+stim\b|from\s+stim\b)", re.MULTILINE)
    offenders = []
    for py in src_root.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        # Strip comments + strings? Naive line-based check is good enough
        # for catching `import stim` / `from stim …`. False positives on
        # the *word* stim in docstrings are tolerable, since they would only
        # be flagged if a docstring happens to contain a literal
        # "import stim" at the start of a line.
        if pattern.search(text):
            offenders.append(str(py.relative_to(src_root.parent)))
    assert not offenders, (
        f"Source-grep found 'import stim' / 'from stim' in: {offenders}. "
        "Stim is forbidden in runtime code (test-time cross-checker only)."
    )

    # __dict__ check (catches module-level imports even if the grep missed)
    leaked = []
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("sparsegf2"):
            continue
        if mod is None or not hasattr(mod, "__dict__"):
            continue
        if "stim" in mod.__dict__:
            leaked.append(mod_name)
    assert not leaked, f"sparsegf2 modules with `stim` in scope (runtime Stim import!): {leaked}"


@pytest.mark.parametrize("attr", ["__version__", "__all__"])
def test_module_metadata_present(attr):
    assert hasattr(sparsegf2, attr)
