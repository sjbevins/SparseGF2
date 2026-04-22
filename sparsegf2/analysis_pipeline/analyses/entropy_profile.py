"""
``entropy_profile`` analysis — S(A_ell) for ell = 1..n/2 at a canonical start.

Writes one fixed-length ``uint16`` array per sample. Canonical starting
position is 0 for MVP; future work can add the four canonical starts
(0, n/4, n/2, 3n/4) and store a ``(4, n/2)`` array per sample.
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


NAME = "entropy_profile"
OUTPUT_FILENAME = "entropy_profile.h5"
OUTPUT_KIND = "h5"
CELL_SCOPE = True
EXPENSIVE = False
DEFAULT_PARAMS: Dict = {"start_position": 0}


def run_cell(ctx: CellContext, params: dict, force: bool = False) -> CellRunResult:
    output = ctx.analysis_dir / OUTPUT_FILENAME
    if output.exists() and not force:
        return CellRunResult(
            name=NAME, status="existing", output_path=output,
            n_samples=ctx.n_samples, params=dict(params),
            message=f"{output.name} already exists (use --force {NAME})",
        )

    merged = {**DEFAULT_PARAMS, **params}
    start = int(merged["start_position"]) % ctx.n

    ctx.analysis_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    n = ctx.n
    half = n // 2
    ell_values = np.arange(1, half + 1, dtype=np.uint16)
    S_of_A = np.zeros((ctx.n_samples, half), dtype=np.uint16)
    for i in range(ctx.n_samples):
        sim = rehydrate_sim(n, ctx.x_stack[i], ctx.z_stack[i])
        for j, ell in enumerate(ell_values):
            qubits = [(start + q) % n for q in range(int(ell))]
            S_of_A[i, j] = int(sim.compute_subsystem_entropy(qubits))

    with h5py.File(output, "w") as f:
        f.attrs["schema_version"] = "1.0.0"
        f.attrs["encoding_version"] = "1.0"
        f.attrs["start_position"] = int(start)
        f.attrs["n"] = int(n)
        f.attrs["n_samples"] = int(ctx.n_samples)
        f.create_dataset("S_of_A", data=S_of_A, chunks=(1, half),
                         compression="gzip", compression_opts=4, shuffle=True)
        f.create_dataset("ell_values", data=ell_values)
        f.create_dataset("sample_seed", data=np.asarray(ctx.seeds, dtype=np.int64))

    runtime = time.perf_counter() - t0
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
