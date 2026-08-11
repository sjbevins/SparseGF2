"""Resolved grids and immutable constants for the PRL production campaign."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CENTERS_CSV = ROOT / "inputs" / "refinement_centers.csv"

PRODUCTION_SIZES = (32, 48, 64, 96, 128, 160, 192, 256)
GRAPH_K = 2
MEAN_DEGREE = 2 * GRAPH_K
TMAX_FACTOR = 8
SCRAMBLE_DEPTH = 32
PRODUCTION_GRAPHS = 500
MASTER_SEED = 3_700_000_001
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CampaignProfile:
    """Fully resolved simulation profile."""

    name: str
    sizes: tuple[int, ...]
    betas: tuple[float, ...]
    p_by_beta: dict[float, tuple[float, ...]]
    n_graphs: int

    @property
    def n_points(self) -> int:
        return sum(len(self.sizes) * len(self.p_by_beta[beta]) for beta in self.betas)

    @property
    def n_trajectories(self) -> int:
        return self.n_points * self.n_graphs


def load_refinement_centers(path: Path = CENTERS_CSV) -> dict[float, float]:
    """Load and validate the beta-dependent steering centers."""
    centers: dict[float, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            beta = float(row["beta"])
            center = float(row["p_center"])
            if not (math.isfinite(beta) and 0.0 <= beta <= 1.0):
                raise ValueError(f"invalid beta in {path}: {beta!r}")
            if not (math.isfinite(center) and 0.0 < center < 1.0):
                raise ValueError(f"invalid p_center in {path}: {center!r}")
            if beta in centers:
                raise ValueError(f"duplicate beta in {path}: {beta:g}")
            centers[beta] = center
    if not centers:
        raise ValueError(f"no refinement centers found in {path}")
    return dict(sorted(centers.items()))


def refinement_p_grid(
    center: float,
    *,
    half_width: float = 0.040,
    step: float = 0.001,
    guard_offsets: tuple[float, ...] = (-0.080, -0.060, 0.060, 0.080),
    p_min: float = 0.005,
    p_max: float = 0.600,
) -> tuple[float, ...]:
    """Return a fine transition grid plus outer phase-bracketing points."""
    n_half = int(round(half_width / step))
    if not math.isclose(n_half * step, half_width, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("half_width must be an integer multiple of step")
    values = {round(center + j * step, 6) for j in range(-n_half, n_half + 1)}
    values.update(round(center + offset, 6) for offset in guard_offsets)
    return tuple(sorted(p for p in values if p_min <= p <= p_max))


def production_profile() -> CampaignProfile:
    """Return the full, fine-grid production profile."""
    centers = load_refinement_centers()
    p_by_beta = {beta: refinement_p_grid(center) for beta, center in centers.items()}
    return CampaignProfile(
        name="production",
        sizes=PRODUCTION_SIZES,
        betas=tuple(centers),
        p_by_beta=p_by_beta,
        n_graphs=PRODUCTION_GRAPHS,
    )


def pilot_profile() -> CampaignProfile:
    """Return a representative, lower-statistics end-to-end pilot."""
    centers = load_refinement_centers()
    requested = (0.0, 0.00328953, 0.01, 0.0316228, 0.1, 1.0)
    p_by_beta = {
        beta: refinement_p_grid(
            centers[beta],
            half_width=0.020,
            step=0.002,
            guard_offsets=(-0.050, 0.050),
        )
        for beta in requested
    }
    return CampaignProfile(
        name="pilot",
        sizes=(32, 64, 128, 256),
        betas=requested,
        p_by_beta=p_by_beta,
        n_graphs=50,
    )


def smoke_profile() -> CampaignProfile:
    """Return a tiny profile that exercises purification and censoring."""
    betas = (0.0, 0.01, 1.0)
    p_values = (0.0, 0.20, 1.0)
    return CampaignProfile(
        name="smoke",
        sizes=(8, 12),
        betas=betas,
        p_by_beta={beta: p_values for beta in betas},
        n_graphs=4,
    )


def get_profile(name: str) -> CampaignProfile:
    """Resolve a named campaign profile."""
    profiles = {
        "smoke": smoke_profile,
        "pilot": pilot_profile,
        "production": production_profile,
    }
    try:
        return profiles[name]()
    except KeyError as exc:
        raise ValueError(f"profile must be one of {sorted(profiles)}; got {name!r}") from exc


def exact_beta(value: float, available: tuple[float, ...]) -> float:
    """Resolve a CLI beta to one stored grid value without silent interpolation."""
    matches = [beta for beta in available if np.isclose(beta, value, rtol=0.0, atol=5e-10)]
    if len(matches) != 1:
        raise ValueError(f"beta={value:g} is not a unique value in the selected profile")
    return matches[0]


__all__ = [
    "CampaignProfile",
    "GRAPH_K",
    "MASTER_SEED",
    "MEAN_DEGREE",
    "PRODUCTION_GRAPHS",
    "PRODUCTION_SIZES",
    "SCHEMA_VERSION",
    "SCRAMBLE_DEPTH",
    "TMAX_FACTOR",
    "exact_beta",
    "get_profile",
    "load_refinement_centers",
    "refinement_p_grid",
]
