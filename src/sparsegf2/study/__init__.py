"""Watts–Strogatz monitored-Clifford sweeps (specification ``docs/misc/watts_strogatz_study_spec_v1_4.md``).

:mod:`.identity` holds the exact tokens of the rates and rewiring probabilities, the seed
hierarchy (graph, run and outcome seeds, stream indices) and the graph records.
``docs/SWEEPS.md`` describes the whole layer.
"""

from sparsegf2.study import identity
from sparsegf2.study.identity import Token

__all__ = ["Token", "identity"]
