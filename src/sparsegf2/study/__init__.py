"""Watts–Strogatz monitored-Clifford sweeps (specification ``docs/misc/watts_strogatz_study_spec_v1_4.md``).

:mod:`.identity` holds the exact tokens of the rates and rewiring probabilities, the seed
hierarchy (graph, run and outcome seeds, stream indices) and the graph records; :mod:`.sweep`
generates a grid of conditions ``(n, beta, p)`` into a directory and reads it back.
``docs/SWEEPS.md`` describes both.
"""

from sparsegf2.study import identity, sweep
from sparsegf2.study.identity import Token
from sparsegf2.study.sweep import SweepSpec

__all__ = ["SweepSpec", "Token", "identity", "sweep"]
