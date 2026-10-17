"""The runner's stored peaks follow the observable catalog (docs/study/04-observables.md §4.6,
Propositions 18(d) and 20; docs/study/05-store-and-reproducibility.md).

* ``peak_D`` records ``se_exact`` True (``D(0)`` from the orderings has its true standard
  errors), on the peak and on every ``sensitivity_z`` entry, and seeds its bootstrap of ``T*``
  with ``observables.bootstrap_seed(key, "peak_D", version)``, the seed the catalog names: a
  bootstrap rebuilt from the stored orderings, exact counts and ``k`` with that seed equals the
  stored one, and the seed of the earlier derivation (``region_seed`` with the horizon) does
  not.
* ``peak_D_J`` records ``se_exact`` from the probe's sampler: False for the swap chain on the
  Watts-Strogatz interaction graph (binomial errors are lower bounds there), True for exact
  samples on a cycle.
* The store's version record carries the bootstrap seed's tag and length, so a store whose
  bootstrap seeds were derived otherwise refuses to be continued by this code.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

import _study_store_support as sup
from sparsegf2.study import observables as ob
from sparsegf2.study.store import STORE_FILE, TableauStore

#: every layer a checkpoint and 40 replicates, so the stored bootstraps have spread
_UNIFORM = {"samples": 12, "exact_budget": 6}
_MANIFEST = dict(
    depth={"layers": {"fixed": 10}, "checkpoints": {"list": [1, 2, 3, 4, 5, 6, 7, 8, 9]}},
    analysis={"peak": {"bootstrap": 40, "z": 1.0}},
    observables={
        **sup.BASE["observables"],
        "uniform_erasure": _UNIFORM,
        "correlated_erasure": [
            # the interaction graph (degree 4): the swap chain
            {"couplings": [0, 1.5], "sizes": [1, 3], "samples": 8, "name": "ring"},
            # the cycle over the qubit labels: exact independent samples
            {"couplings": [0.5], "sizes": "1:6", "samples": 8, "geometry": "cycle", "name": "loop"},
        ],
    },
)


def _params(observable: str, rule: dict, **extra) -> dict:
    names = ob.get(observable).params
    return {**{k: v for k, v in rule.items() if k in names}, **extra}


def _rebuilt_bootstrap(store: TableauStore, key, data: dict, seed) -> dict:
    """The bootstrap of ``T*`` rebuilt from the stored records as the catalog says: the stored
    orderings and exact counts at the domain's depths, ``k`` there, the horizon, and ``seed``."""
    domain = data["domain"]
    rows = [store.observation(key, t, "uniform_failure", params=_UNIFORM)["data"] for t in domain]
    k_at = [int(store.observation(key, t, "k")["value"]) for t in domain]
    boot = ob.bootstrap_peak(
        domain,
        np.array([r["first_failure"] for r in rows]),
        key.n,
        z=data["z"],
        replicates=data["bootstrap"]["replicates"],
        seed=seed,
        exact_counts=np.array([r["exact_counts"] for r in rows]),
        k_at=k_at,
        horizon=data["horizon"],
    )
    lo, hi = boot.interval(0.95)
    finite = boot.t_star[np.isfinite(boot.t_star)]
    return {
        "interval95": [lo, hi],
        "missing_fraction": boot.missing_fraction,
        "horizon_fraction": boot.horizon_fraction,
        "sd": float(finite.std(ddof=1)) if finite.size > 1 else None,
    }


def _summary(stored: dict) -> dict:
    return {k: stored[k] for k in ("interval95", "missing_fraction", "horizon_fraction", "sd")}


def test_stored_peaks_record_their_errors_and_the_catalog_bootstrap_seed(tmp_path):
    m = sup.manifest(**_MANIFEST)
    rule = m.analysis.peak.to_dict()
    with sup.build(tmp_path / "s", m) as store:
        keys = store.trajectory_keys()
        assert len(keys) == 12
        version = ob.get("peak_D").version
        old_differs = 0
        for key in keys:
            peak_d = store.observation(key, -1, "peak_D", params=_params("peak_D", rule))
            data = peak_d["data"]
            # D(0) from the orderings: true standard errors, on the peak and every z
            assert data["se_exact"] is True
            assert [s["se_exact"] for s in data["sensitivity"]] == [True] * 3
            # the catalog's seed: the key, the id and the version (Proposition 18(d))
            seed = ob.bootstrap_seed(key, "peak_D", version)
            assert data["bootstrap"]["seed"] == list(seed)
            assert len(seed) == ob.BOOTSTRAP_SEED_WORDS and seed[0] == ob.BOOTSTRAP_SEED_TAG
            # following the catalog reproduces the stored bootstrap exactly
            assert _rebuilt_bootstrap(store, key, data, seed) == _summary(data["bootstrap"])
            old = ob.region_seed(key, data["horizon"], 0, 0.0, "peak_D", version)
            old_differs += _rebuilt_bootstrap(store, key, data, old) != _summary(data["bootstrap"])
            # D(J): the chain on the interaction graph has lower-bound errors, the cycle exact ones
            for coupling, probe, exact in ((0.0, "ring", False), (1.5, "ring", False), (0.5, "loop", True)):  # fmt: skip
                peak_j = store.observation(
                    key, -1, "peak_D_J", params=_params("peak_D_J", rule, J=coupling)
                )
                assert peak_j["data"]["probe"] == probe
                assert peak_j["data"]["se_exact"] is exact, (key, coupling)
                assert [s["se_exact"] for s in peak_j["data"]["sensitivity"]] == [exact] * 3
        # the data discriminate between the seeds: the earlier derivation gives another
        # bootstrap (all 12 here; 40 replicates with spread; 0 would mean no test)
        assert old_differs >= 6, old_differs


def test_the_version_record_carries_the_bootstrap_seed(tmp_path):
    path = tmp_path / "s"
    with sup.build(path, run=False) as store:
        versions = store.meta()["versions"]
    assert versions["bootstrap_seed_tag"] == ob.BOOTSTRAP_SEED_TAG == 0x57535242
    assert versions["bootstrap_seed_words"] == ob.BOOTSTRAP_SEED_WORDS == 15
    # a store made before the bootstrap seed was recorded (its peaks seeded otherwise) is read,
    # but not continued
    earlier = {k: v for k, v in versions.items() if not k.startswith("bootstrap_seed")}
    with sqlite3.connect(path / STORE_FILE) as con:
        con.execute(
            "UPDATE meta SET value = ? WHERE key = 'versions'",
            (json.dumps(earlier, sort_keys=True, separators=(",", ":")),),
        )
    con.close()
    with TableauStore.open(path) as reader:
        assert "bootstrap_seed_tag" not in reader.meta()["versions"]
    with pytest.raises(ValueError, match="bootstrap_seed_tag"):
        TableauStore.open(path, write=True)
