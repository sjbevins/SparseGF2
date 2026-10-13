"""Exceptions raised by sparsegf2_v3."""

from __future__ import annotations

__all__ = ["PhaseFreeError"]


class PhaseFreeError(RuntimeError):
    """A quantity that needs sign (phase) tracking was requested from a phase-free simulator.

    Phase-free simulators (``Simulator(n, phases=False)``) store only the
    unsigned stabilizer group. That determines every entanglement quantity
    exactly, but not measurement outcomes, expectation values or signs. Use
    ``phases=True`` for those.
    """
