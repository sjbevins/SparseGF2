"""
``weight_spectrum`` analysis — per-generator weights for every sample.

Writes a ``uint16[S, 2n]`` matrix to ``analysis/weight_spectrum.h5``. Each row
is the weight vector (``supp_len``) for one sample's 2n stabilizer generators.
This is the full per-generator spectrum, not just the moments — the moments
already live in ``weight_stats.parquet``.

Flagged ``EXPENSIVE`` in the sense of *storage* more than compute: at n=512
with S=1500 samples per cell the array is 1.5 MB per cell before compression.
Still small in absolute terms, but opt-outable for the use-case where the
researcher doesn't need the full spectrum.
"""
from __future__ import annotations

import time
from typing import Dict

import h5py
import numpy as np

from sparsegf2 import __version__ as _SGF2_VERSION
from sparsegf2.analysis_pipeline.analyses._common import CellContext, CellRunResult
from sparsegf2.analysis_pipeline.rehydrate import rehydrate_sim
from sparsegf2.analysis_pipeline.registry import (
    make_entry, read_cell_registry, write_cell_registry,
)


NAME = "weight_spectrum"
OUTPUT_FILENAME = "weight_spectrum.h5"
OUTPUT_KIND = "h5"
CELL_SCOPE = True
EXPENSIVE = True
DEFAULT_PARAMS: Dict = {}


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

    n = ctx.n
    N = 2 * n
    weights = np.zeros((ctx.n_samples, N), dtype=np.uint16)
    for i in range(ctx.n_samples):
        sim = rehydrate_sim(n, ctx.x_stack[i], ctx.z_stack[i])
        # supp_len[r] is the number of non-identity qubits in generator r.
        weights[i, :] = np.asarray(sim.supp_len[:N], dtype=np.uint16)

    with h5py.File(output, "w") as f:
        f.attrs["schema_version"] = "1.0.0"
        f.attrs["encoding_version"] = "1.0"
        f.attrs["n"] = int(n)
        f.attrs["n_samples"] = int(ctx.n_samples)
        f.attrs["row_layout"] = "rows 0..n-1 = destabilizers, n..2n-1 = stabilizers"
        f.create_dataset("weights", data=weights, chunks=(1, N),
                         compression="gzip", compression_opts=4, shuffle=True)
        f.create_dataset("sample_seed", data=np.asarray(ctx.seeds, dtype=np.int64))

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
