"""
``weight_stats`` analysis — full per-sample weight-distribution moments.

Wraps :func:`sparsegf2.analysis.observables.observe` so every scalar that
function currently returns is persisted one-row-per-sample in
``analysis/weight_stats.parquet``. Adding a new scalar to ``observe()`` in
the future simply means adding the column here.
"""
from __future__ import annotations

import time
from typing import Dict

import pyarrow as pa
import pyarrow.parquet as pq

from sparsegf2 import __version__ as _SGF2_VERSION
from sparsegf2.analysis.observables import observe
from sparsegf2.analysis_pipeline.analyses._common import CellContext, CellRunResult
from sparsegf2.analysis_pipeline.rehydrate import rehydrate_sim
from sparsegf2.analysis_pipeline.registry import (
    make_entry, read_cell_registry, write_cell_registry,
)


NAME = "weight_stats"
OUTPUT_FILENAME = "weight_stats.parquet"
OUTPUT_KIND = "parquet"
CELL_SCOPE = True
EXPENSIVE = False
DEFAULT_PARAMS: Dict = {}


# Canonical list of scalar fields returned by observe(sim). Keep this list
# sorted so the Parquet schema is stable across versions.
_OBSERVE_FIELDS = [
    ("a_max", pa.float64()),
    ("a_min", pa.float64()),
    ("abar", pa.float64()),
    ("cv_a", pa.float64()),
    ("cv_w", pa.float64()),
    ("identity_holds", pa.bool_()),
    ("mean_wt_x", pa.float64()),
    ("n_anticommuting_qubits", pa.int64()),
    ("pei_proxy", pa.float64()),
    ("pivot_speedup", pa.float64()),
    ("skew_a", pa.float64()),
    ("skew_w", pa.float64()),
    ("std_a", pa.float64()),
    ("std_w", pa.float64()),
    ("var_a", pa.float64()),
    ("var_w", pa.float64()),
    ("w_max", pa.float64()),
    ("w_min", pa.float64()),
    ("wbar", pa.float64()),
    ("wbar_destab", pa.float64()),
    ("wbar_stab", pa.float64()),
    ("wcp_proxy", pa.float64()),
    ("weight_mass", pa.float64()),
]

_SCHEMA = pa.schema(
    [("sample_seed", pa.int64())]
    + _OBSERVE_FIELDS
    + [("runtime_s", pa.float64())]
)

_FIELD_NAMES = [f.name for f in _SCHEMA]


def _cast_value(col: str, value):
    field = _SCHEMA.field(col)
    if field.type == pa.bool_():
        return bool(value)
    if field.type in (pa.int32(), pa.int64()):
        return int(value)
    return float(value)


def run_cell(ctx: CellContext, params: dict, force: bool = False) -> CellRunResult:
    output = ctx.analysis_dir / OUTPUT_FILENAME
    if output.exists() and not force:
        return CellRunResult(
            name=NAME, status="existing", output_path=output,
            n_samples=ctx.n_samples, params=dict(params),
            message=f"{output.name} already exists (use --force {NAME})",
        )

    ctx.analysis_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    data: Dict[str, list] = {name: [] for name in _FIELD_NAMES}
    n = ctx.n
    for i in range(ctx.n_samples):
        t_s = time.perf_counter()
        sim = rehydrate_sim(n, ctx.x_stack[i], ctx.z_stack[i])
        row = observe(sim)
        rt = time.perf_counter() - t_s

        data["sample_seed"].append(int(ctx.seeds[i]))
        for col, _ in _OBSERVE_FIELDS:
            data[col].append(_cast_value(col, row[col]))
        data["runtime_s"].append(float(rt))

    table = pa.Table.from_pydict(data, schema=_SCHEMA)
    pq.write_table(table, output)
    runtime = time.perf_counter() - t0

    merged = {**DEFAULT_PARAMS, **params}
    entries = read_cell_registry(ctx.analysis_dir)
    entries[NAME] = make_entry(
        package_version=_SGF2_VERSION, params=merged,
        runtime_s=runtime, n_samples=ctx.n_samples,
    ).__dict__
    write_cell_registry(ctx.analysis_dir, entries)

    return CellRunResult(
        name=NAME, status="computed", output_path=output,
        runtime_s=runtime, n_samples=ctx.n_samples, params=merged,
    )
