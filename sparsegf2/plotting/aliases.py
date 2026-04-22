"""
Derived-column aliases for :func:`plot_vs_p` and future primitives.

An alias expands at load time into a :class:`polars.Expr` that synthesizes
the desired column from already-present raw columns. This keeps the primitive
API stringly-typed (``y="k_over_n"``) while avoiding the ambiguity of
evaluating arbitrary user expressions.

Add a new alias by updating :data:`DERIVED_ALIASES`. No primitive code has
to change.
"""
from __future__ import annotations

from typing import Callable, Dict

import polars as pl


# Each alias maps a short name to a callable taking no arguments and
# returning a polars expression that can be .alias()-renamed to the short name.
DERIVED_ALIASES: Dict[str, Callable[[], pl.Expr]] = {
    "k_over_n": lambda: (pl.col("obs.k").cast(pl.Float64)
                         / pl.col("n").cast(pl.Float64)),
    "d_cont_over_n": lambda: (pl.col("distances.d_cont").cast(pl.Float64)
                              / pl.col("n").cast(pl.Float64)),
    "d_min_over_n": lambda: (pl.col("distances.d_min").cast(pl.Float64)
                             / pl.col("n").cast(pl.Float64)),
    "d_logical_min_over_n": lambda: (pl.col("logical_weights.d_logical_min").cast(pl.Float64)
                                     / pl.col("n").cast(pl.Float64)),
}


def resolve_alias(name: str) -> "pl.Expr | None":
    """Return the polars expression for ``name`` if aliased, else ``None``."""
    factory = DERIVED_ALIASES.get(name)
    if factory is None:
        return None
    return factory().alias(name)


__all__ = ["DERIVED_ALIASES", "resolve_alias"]
