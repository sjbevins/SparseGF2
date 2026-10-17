"""The Watts–Strogatz monitored-Clifford study pipeline (specification v1.4).

``docs/misc/watts_strogatz_study_spec_v1_4.md`` is the scientific specification and
``docs/specs/2026-10-17-ws-study-pipeline-design.md`` the design: experiment manifests,
trajectory identities and seeds (:mod:`.identity`, :mod:`.manifest`), the indexed tableau
store (:mod:`.store`, :mod:`.runner`), the observable catalog (:mod:`.observables`), nested
ensemble statistics (:mod:`.stats`), finite-size scaling (:mod:`.fss`), the JSON service layer
that the ``sparsegf2 study`` commands and the MCP ``study_*`` tools share (:mod:`.api`), and
the diagnostic figures (:mod:`.plots`; matplotlib is imported only when a figure is drawn).

The usual entry points are the service functions (``from sparsegf2.study import api``;
``api.init``, ``api.plan``, ``api.run``, ``api.status``, ``api.aggregate``, ``api.fss``, ...)
and, for Python work on the stored data, :class:`TableauStore`.
"""

from sparsegf2.study import (
    api,
    fss,
    identity,
    manifest,
    observables,
    plots,
    runner,
    stats,
    store,
)
from sparsegf2.study.identity import Token, TrajectoryKey
from sparsegf2.study.manifest import ExperimentManifest
from sparsegf2.study.store import TableauStore

__all__ = [
    "ExperimentManifest",
    "TableauStore",
    "Token",
    "TrajectoryKey",
    "api",
    "fss",
    "identity",
    "manifest",
    "observables",
    "plots",
    "runner",
    "stats",
    "store",
]
