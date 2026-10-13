"""Results returned by :meth:`Simulator.run` and :func:`sample`."""

from __future__ import annotations

import numpy as np

from sparsegf2_v3._compile import Program
from sparsegf2_v3.errors import PhaseFreeError

__all__ = ["RunResult", "SampleResult"]


def _phase_free_error(what: str) -> PhaseFreeError:
    return PhaseFreeError(
        f"{what} needs measurement outcomes, which do not exist in phase-free mode. "
        "Run with phases=True (the default)."
    )


class RunResult:
    """What one run of a circuit produced.

    Attributes
    ----------
    random : ndarray of bool
        ``random[k]`` is True if measurement ``k`` had a random outcome. This is
        a physical property of the trajectory and is available in both modes.
    observations : list
        ``(tick, value)`` pairs from the ``observe`` callback, if one was given.
    """

    __slots__ = ("_phases", "_program", "_record", "observations", "random")

    def __init__(self, phases: bool, program: Program, record, random, observations):
        self._phases = phases
        self._program = program
        self._record = record
        self.random = random.astype(bool)
        self.observations = observations

    @property
    def measurements(self) -> np.ndarray:
        """The measurement record (uint8, one entry per measurement result)."""
        if not self._phases:
            raise _phase_free_error("RunResult.measurements")
        return self._record

    @property
    def detectors(self) -> np.ndarray:
        """Detection events, in Stim's convention.

        Each DETECTOR's parity XOR its value in the noiseless reference sample
        (every collapse biased to +1), so a noiseless run gives all zeros for
        properly defined detectors, exactly like Stim's samplers. For a
        detector whose parity is random even without noise, the reference can
        differ from Stim's (see ``Program.reference_record``): same
        distribution, possibly different per-shot values.
        """
        if not self._phases:
            raise _phase_free_error("RunResult.detectors")
        return self._program.detection_events(self._record)

    @property
    def observables(self) -> np.ndarray:
        """Observable flips: each OBSERVABLE_INCLUDE parity XOR its reference value (Stim's convention)."""
        if not self._phases:
            raise _phase_free_error("RunResult.observables")
        return self._program.observable_flips(self._record)

    def __repr__(self) -> str:
        return (
            f"RunResult(measurements={self._program.num_measurements}, "
            f"random={int(self.random.sum())}, phases={self._phases})"
        )


class SampleResult:
    """Many independent shots of one circuit (from :func:`sparsegf2_v3.sample`).

    Arrays have one row per shot. ``random`` is available in both modes;
    ``measurements``, ``detectors`` and ``observables`` need phases.
    """

    __slots__ = ("_phases", "_program", "_records", "random")

    def __init__(self, phases: bool, program: Program, records, random):
        self._phases = phases
        self._program = program
        self._records = records
        self.random = random.astype(bool)

    @property
    def shots(self) -> int:
        return self.random.shape[0]

    @property
    def measurements(self) -> np.ndarray:
        if not self._phases:
            raise _phase_free_error("SampleResult.measurements")
        return self._records

    @property
    def detectors(self) -> np.ndarray:
        """Detection events per shot (Stim's convention; see :attr:`RunResult.detectors`)."""
        if not self._phases:
            raise _phase_free_error("SampleResult.detectors")
        return self._program.detection_events(self._records)

    @property
    def observables(self) -> np.ndarray:
        """Observable flips per shot (Stim's convention)."""
        if not self._phases:
            raise _phase_free_error("SampleResult.observables")
        return self._program.observable_flips(self._records)

    def __repr__(self) -> str:
        return (
            f"SampleResult(shots={self.shots}, measurements={self._program.num_measurements}, "
            f"phases={self._phases})"
        )
