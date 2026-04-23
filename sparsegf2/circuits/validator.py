"""
Pre-flight compatibility validator.

Before any simulation work begins, the driver calls :func:`validate_config` to
check every ``(n, matching_mode)`` pair in the sweep. If any pair is
incompatible with the requested graph, the validator builds a human-readable
report enumerating the problems and the set of modes that *would* work, then
raises :class:`CompatibilityError`. The full run aborts with exit code 2 at
the CLI level before any sample is written.

The rules are simple for the MVP graphs (cycle, complete):

- ``round_robin`` / ``palette``: graph must admit a 1-factorization
  -> cycle or complete with even ``n``.
- ``fresh``: graph must have at least one perfect matching
  -> cycle or complete with even ``n``.

Stochastic graphs (deferred) will extend this with a probabilistic check
("perfect matching exists in X/Y sampled realizations").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from sparsegf2.circuits.config import RunConfig
from sparsegf2.circuits.graphs import GraphTopology, parse_graph_spec
from sparsegf2.circuits.matching import MATCHING_MODES, available_modes


class CompatibilityError(RuntimeError):
    """Raised by :func:`validate_config` when the sweep contains an incompatible
    ``(graph, matching_mode, n)`` triple."""

    def __init__(self, report: "ValidationReport") -> None:
        self.report = report
        super().__init__(report.format())


@dataclass
class ValidationReport:
    """Outcome of a pre-flight validation pass.

    Attributes
    ----------
    graph_spec : str
        The graph spec the sweep requested.
    matching_mode : str
        The matching mode the sweep requested.
    sizes : list of int
        The sizes the sweep would run.
    ok : list of int
        Sizes for which ``(graph_spec, matching_mode, n)`` is compatible.
    incompatible : list of (int, str)
        ``(n, reason)`` pairs for each size where the mode is incompatible.
    available_modes_by_n : dict
        ``n -> list of mode names`` available at that size.
    """

    graph_spec: str
    matching_mode: str
    sizes: List[int]
    ok: List[int] = field(default_factory=list)
    incompatible: List[Tuple[int, str]] = field(default_factory=list)
    available_modes_by_n: Dict[int, List[str]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.incompatible

    # --------------------------------------------------------------

    def format(self) -> str:
        """Render the report as the human-readable error text used at the CLI.

        Uses only ASCII symbols so that the message is readable on any
        platform, including Windows consoles with cp1252 codepage.
        """
        lines: List[str] = []
        lines.append("PRE-FLIGHT VALIDATION FAILED")
        lines.append("")
        lines.append("Config:")
        lines.append(f"  graph_spec      = {self.graph_spec}")
        lines.append(f"  matching_mode   = {self.matching_mode}")
        lines.append(f"  sizes           = {self.sizes}")
        lines.append("")
        lines.append("Incompatible cells:")
        for n, reason in self.incompatible:
            lines.append(f"  n={n}  [FAIL]  {reason}")
        lines.append("")
        lines.append("Available modes for each size:")
        for n in self.sizes:
            if n in self.incompatible_sizes:
                modes = self.available_modes_by_n.get(n, [])
                if modes:
                    lines.append(f"  n={n}  ({', '.join(modes)})")
                else:
                    lines.append(f"  n={n}  (none -- no perfect matching)")
            else:
                modes = self.available_modes_by_n.get(n, [])
                # Passing sizes always have non-empty modes (validate_config
                # only appends to ok_sizes when the requested mode is in
                # modes_here). Report [ok] unconditionally.
                lines.append(f"  n={n}  [ok]    " + ", ".join(modes))
        lines.append("")
        lines.append("Suggested fixes:")
        lines.append("  (a) remove the offending sizes from --sizes")
        lines.append("  (b) switch to a matching_mode listed as available above")
        lines.append("  (c) switch to a gating mode that does not require perfect")
        lines.append("      matchings (not yet implemented in MVP)")
        lines.append("")
        lines.append("Aborting before any samples are run.")
        return "\n".join(lines)

    @property
    def incompatible_sizes(self) -> List[int]:
        return [n for n, _ in self.incompatible]


# Public API

def validate_config(cfg: RunConfig) -> ValidationReport:
    """Validate a :class:`RunConfig` against its graph and matching mode.

    Parameters
    ----------
    cfg : RunConfig
        The sweep configuration.

    Returns
    -------
    ValidationReport
        A fully-populated report. ``report.passed`` is ``True`` iff every
        ``(graph, matching_mode, n)`` triple is compatible.

    Raises
    ------
    CompatibilityError
        If any incompatibility is found. The exception's message is exactly
        what :meth:`ValidationReport.format` returns.
    """
    graph_spec = cfg.circuit.graph_spec
    mode = cfg.circuit.matching_mode
    gating = cfg.circuit.gating_mode
    report = ValidationReport(
        graph_spec=graph_spec, matching_mode=mode, sizes=list(cfg.sizes)
    )

    for n in cfg.sizes:
        graph = parse_graph_spec(graph_spec, n, seed=cfg.circuit.base_seed)
        modes_here = available_modes(graph)
        report.available_modes_by_n[n] = modes_here
        if gating == "random_edge":
            # random_edge only needs at least one edge in the graph;
            # matching compatibility is irrelevant.
            if len(graph.edges) == 0:
                report.incompatible.append((n, f"{graph.name}: no edges"))
            else:
                report.ok.append(n)
        elif mode in modes_here:
            report.ok.append(n)
        else:
            reason = _reason_for_incompatibility(graph, mode)
            report.incompatible.append((n, reason))

    if not report.passed:
        raise CompatibilityError(report)
    return report


def _reason_for_incompatibility(graph: GraphTopology, mode: str) -> str:
    """Produce a one-line explanation of why ``mode`` fails for ``graph``."""
    if mode not in MATCHING_MODES:
        return f"unknown matching_mode {mode!r}"
    if mode in ("round_robin", "palette"):
        if graph.one_factorization is None:
            return (
                f"{graph.name}: no 1-factorization available "
                f"(graph is not class-1 regular, or n is odd)"
            )
        return f"{graph.name}: internal error — 1-factorization present but mode rejected"
    if mode == "fresh":
        if graph.fresh_matching_sampler is None:
            return f"{graph.name}: no perfect matching exists"
        return f"{graph.name}: internal error — perfect-matching sampler present but mode rejected"
    return f"{graph.name}: mode {mode!r} is not recognized"


__all__ = [
    "CompatibilityError",
    "ValidationReport",
    "validate_config",
]
