"""The expurgation loop: sample, extract, re-validate, measure, stop.

One round of the algorithm (Gullans et al., PRX 11, 031066 (2021),
Sec. VI):

1. sample an erasure pattern from the target error model;
2. extract the zero-syndrome, nontrivially-logical operators supported
   on it (:func:`sparsegf2.expurgation.expurgation_candidates`),
   lightest first;
3. re-validate each candidate against the *current* tableau just before
   measuring (earlier measurements in the same round may have made it
   detectable or trivial) and measure the survivors, spending one
   logical pair each;
4. repeat with fresh patterns until the rate target is reached, the
   validation metric passes, candidates dry up, or the code dies
   (``k == 0``).

Skipping a stale candidate changes nothing, and measuring a validated
one spends exactly one logical pair, so ``k`` falls by precisely the
number of measurements performed; distance and optimal-recovery
probability are non-decreasing throughout (the paper's Propositions 1
and 2).

Validation patterns are frozen once at the start, so the before/after
recovery numbers (and any ``recovery_target`` check) compare the same
error events; per pattern the recovery is monotone under expurgation,
which makes the target check stable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from sparsegf2.errors import InvalidArgumentError
from sparsegf2.expurgation.erasure import (
    expurgation_candidates,
    recovery_probability,
    sample_erasure,
)
from sparsegf2.expurgation.roles import STRATEGIES, StabilizerCode, _exact_integer_array


@dataclass(frozen=True)
class ExpurgationConfig:
    """Configuration for :func:`expurgate`.

    Parameters
    ----------
    strategy
        ``"gauge"`` (default; measured operators become gauge pairs and
        the original checks are untouched, the paper's preferred
        variant) or ``"stabilizer"`` (measured operators become new
        checks).
    erasure_rate, erasure_count
        The erasure model: give exactly one. ``erasure_rate`` erases
        each site independently with that probability;
        ``erasure_count`` erases a uniform random subset of exactly
        that many sites.
    sites
        Optional pool of erasable sites (defaults to every qubit). Use
        this to target the error model at a subset, for example the
        system half of a purification tableau.
    k_target
        Stop once ``code.k`` is at or below this. ``0`` runs until one
        of the other criteria fires.
    recovery_target
        Optional: stop once the mean recovery probability over the
        frozen validation patterns reaches this value. Requires
        ``validation_patterns > 0``.
    max_rounds
        Hard cap on the number of erasure patterns drawn.
    max_barren_rounds
        Stop after this many consecutive rounds in which no candidate
        survived re-validation (the steady state of the paper's loop).
    validation_patterns
        Number of frozen erasure patterns used for the recovery metric
        (before, after, and the optional per-round target check). ``0``
        disables validation entirely.
    seed
        Seed for the driver's random generator (erasure sampling and
        validation patterns). ``None`` gives a fresh nondeterministic
        generator.
    """

    strategy: str = "gauge"
    erasure_rate: float | None = None
    erasure_count: int | None = None
    sites: tuple[int, ...] | None = None
    k_target: int = 0
    recovery_target: float | None = None
    max_rounds: int = 64
    max_barren_rounds: int = 3
    validation_patterns: int = 32
    seed: int | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous and lossy settings before a run mutates a code."""
        if self.strategy not in STRATEGIES:
            raise InvalidArgumentError(f"strategy={self.strategy!r} not in {STRATEGIES}")
        if self.erasure_rate is not None and (
            not isinstance(self.erasure_rate, Real)
            or isinstance(self.erasure_rate, (bool, np.bool_))
            or not math.isfinite(float(self.erasure_rate))
            or not 0.0 <= float(self.erasure_rate) <= 1.0
        ):
            raise InvalidArgumentError(
                f"erasure_rate must be a finite real number in [0, 1], got {self.erasure_rate!r}"
            )
        integer_fields = (
            ("erasure_count", self.erasure_count, 0),
            ("k_target", self.k_target, 0),
            ("max_rounds", self.max_rounds, 0),
            ("max_barren_rounds", self.max_barren_rounds, 1),
            ("validation_patterns", self.validation_patterns, 0),
        )
        for name, value, minimum in integer_fields:
            if value is None and name == "erasure_count":
                continue
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise InvalidArgumentError(
                    f"{name} must be an exact integer >= {minimum}, got {value!r}"
                )
            object.__setattr__(self, name, int(value))
        if self.recovery_target is not None:
            if (
                not isinstance(self.recovery_target, Real)
                or isinstance(self.recovery_target, (bool, np.bool_))
                or not math.isfinite(float(self.recovery_target))
                or not 0.0 <= float(self.recovery_target) <= 1.0
            ):
                raise InvalidArgumentError(
                    "recovery_target must be a finite real number in [0, 1], "
                    f"got {self.recovery_target!r}"
                )
            object.__setattr__(self, "recovery_target", float(self.recovery_target))
        if self.seed is not None:
            if (
                isinstance(self.seed, (bool, np.bool_))
                or not isinstance(self.seed, (int, np.integer))
                or self.seed < 0
            ):
                raise InvalidArgumentError(
                    f"seed must be None or a non-negative exact integer, got {self.seed!r}"
                )
            object.__setattr__(self, "seed", int(self.seed))
        if self.sites is not None:
            sites = _exact_integer_array(self.sites, name="sites")
            if sites.size and (sites < 0).any():
                raise InvalidArgumentError("sites must contain non-negative indices")
            if np.unique(sites).shape[0] != sites.shape[0]:
                raise InvalidArgumentError("sites must not repeat")
            object.__setattr__(self, "sites", tuple(int(site) for site in sites))


@dataclass
class ExpurgationResult:
    """Record of one :func:`expurgate` run.

    ``k_trajectory[r]`` is ``code.k`` after round ``r``;
    ``measured_weights`` / ``measured_pairs`` / ``measured_rounds`` are
    aligned per measurement. ``stop_reason`` is one of ``"k_zero"``
    (the code died, expurgation failed), ``"k_target"``,
    ``"recovery_target"``, ``"barren"``, or ``"max_rounds"``.
    ``recovery_before`` / ``recovery_after`` are the mean recovery over
    the frozen validation patterns, or ``None`` when validation was
    disabled.
    """

    k_initial: int
    k_final: int
    rounds: int
    stop_reason: str
    k_trajectory: list[int] = field(default_factory=list)
    measured_weights: list[int] = field(default_factory=list)
    measured_pairs: list[int] = field(default_factory=list)
    measured_rounds: list[int] = field(default_factory=list)
    recovery_before: float | None = None
    recovery_after: float | None = None


def mean_recovery(code: StabilizerCode, patterns: list[NDArray[np.int64]]) -> float:
    """Mean exact recovery probability over a list of erasure patterns.

    The validation metric of the loop:
    :math:`\\overline{P(R)} = \\mathbb{E}_e\\, 2^{-r_M(e)}`, evaluated
    exactly (two GF(2) ranks per pattern). Returns ``1.0`` for an empty
    list (nothing to fail on).
    """
    if not patterns:
        return 1.0
    return float(np.mean([recovery_probability(code, e) for e in patterns]))


def expurgate(code: StabilizerCode, config: ExpurgationConfig | None = None) -> ExpurgationResult:
    """Run the expurgation loop on ``code`` in place.

    Mutates ``code`` (its simulator and role array); snapshot with
    ``code.copy()`` first if the original is still needed. Returns the
    :class:`ExpurgationResult` record.
    """
    cfg = ExpurgationConfig() if config is None else config
    if cfg.strategy not in STRATEGIES:
        raise InvalidArgumentError(f"strategy={cfg.strategy!r} not in {STRATEGIES}")
    if (cfg.erasure_rate is None) == (cfg.erasure_count is None):
        raise InvalidArgumentError("give exactly one of erasure_rate= or erasure_count=")
    if cfg.k_target < 0:
        raise InvalidArgumentError(f"k_target must be non-negative, got {cfg.k_target}")
    if cfg.recovery_target is not None and cfg.validation_patterns <= 0:
        raise InvalidArgumentError("recovery_target requires validation_patterns > 0")
    rng = np.random.default_rng(cfg.seed)
    sites = None if cfg.sites is None else np.asarray(cfg.sites, dtype=np.int64)

    def draw() -> NDArray[np.int64]:
        return sample_erasure(
            code.n, rng, rate=cfg.erasure_rate, count=cfg.erasure_count, sites=sites
        )

    patterns = [draw() for _ in range(cfg.validation_patterns)]
    result = ExpurgationResult(
        k_initial=code.k,
        k_final=code.k,
        rounds=0,
        stop_reason="max_rounds",
        recovery_before=mean_recovery(code, patterns) if patterns else None,
    )

    barren = 0
    while True:
        if code.k == 0:
            result.stop_reason = "k_zero"
            break
        if code.k <= cfg.k_target and cfg.k_target > 0:
            result.stop_reason = "k_target"
            break
        if cfg.recovery_target is not None and (
            mean_recovery(code, patterns) >= cfg.recovery_target
        ):
            result.stop_reason = "recovery_target"
            break
        if barren >= cfg.max_barren_rounds:
            result.stop_reason = "barren"
            break
        if result.rounds >= cfg.max_rounds:
            result.stop_reason = "max_rounds"
            break

        erased = draw()
        measured_this_round = 0
        for cand in expurgation_candidates(code, erased):
            if code.k <= cfg.k_target and cfg.k_target > 0:
                break
            if code.k == 0:
                break
            # Mid-sequence re-validation: earlier measurements in this
            # round may have made the candidate detectable (a success,
            # skip it as an error the code now catches) or trivial.
            syndrome, _logical = code.commutation_bits(cand.qubits, cand.letters)
            if syndrome.any():
                continue
            pair = code.measure(cand.qubits, cand.letters, strategy=cfg.strategy)
            if pair is None:
                continue
            measured_this_round += 1
            result.measured_weights.append(cand.weight)
            result.measured_pairs.append(pair)
            result.measured_rounds.append(result.rounds)
        result.rounds += 1
        result.k_trajectory.append(code.k)
        barren = barren + 1 if measured_this_round == 0 else 0

    result.k_final = code.k
    result.recovery_after = mean_recovery(code, patterns) if patterns else None
    return result
