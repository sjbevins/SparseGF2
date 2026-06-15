#!/usr/bin/env python
"""Run a chunked, resumable parameter sweep as a cluster job array.

Each array task computes a disjoint stride-slice of the chunk list, so K array
tasks spread across K nodes cover the whole study in parallel, and any task can
be re-queued without redoing finished work (each chunk is a resumable Study on
disk: a chunk whose rows.parquet exists is skipped). Within a task the sweep
fills every allocated core with n_jobs=-1, which resolves through the scheduler
allocation rather than the whole node.

Adapt CHUNKS and base_config to your study, then submit with the companion
hpc_sweep_array.sbatch, or directly:

    sbatch --array=0-7 --cpus-per-task=32 scripts/hpc_sweep_array.sbatch
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from sparsegf2.analysis import Study, available_cores
from sparsegf2.circuits import CircuitConfig

OUT = Path(os.environ.get("STUDY_OUT", "runs/hpc_study"))

# --- the work list: one resumable Study per entry -------------------------
N_VALS = [64, 128, 256, 512]
P_VALS = list(np.round(np.linspace(0.08, 0.45, 16), 3))
SEEDS = {64: 400, 128: 300, 256: 200, 512: 100}
PICTURES = ["purification", "single_ref"]
CHUNKS = [(picture, n) for picture in PICTURES for n in N_VALS]


def base_config(n: int, picture: str) -> CircuitConfig:
    return CircuitConfig(
        graph_spec="complete",
        n=n,
        p=P_VALS[0],
        picture=picture,
        gating_mode="brickwork",
        matching_mode="palette",
        measurement_mode="bernoulli",
        depth_mode="until_purified",
        depth_factor=40,
        hybrid=True,
    )


def run_chunk(picture: str, n: int) -> None:
    d = OUT / f"{picture}_n{n}"
    if (d / "rows.parquet").exists():
        print(f"[skip] {d} already complete", flush=True)
        return
    print(f"[run]  {d}  ({len(P_VALS) * SEEDS[n]} cells) on {available_cores()} cores", flush=True)
    Study.run(
        d,
        base_config(n, picture),
        grid={"n": [n], "p": P_VALS},
        seeds=SEEDS[n],
        save_tableaux=False,
        n_jobs=-1,
        progress=False,
    )


def main() -> None:
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    # Normalize the task id to a contiguous 0-based rank so the stride slice
    # partitions the work correctly even for a non-zero-based or stepped array
    # (e.g. --array=5-12 or --array=0-10:2). Without this, CHUNKS[task::count]
    # would silently drop or duplicate chunks for any array that is not 0..N-1.
    task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", str(task)))
    step = int(os.environ.get("SLURM_ARRAY_TASK_STEP", "1")) or 1
    rank = (task - task_min) // step
    mine = CHUNKS[rank::count]  # this task's stride-slice of the work list
    print(
        f"array rank {rank}/{count} (task id {task}): {len(mine)} of {len(CHUNKS)} chunk(s)",
        flush=True,
    )
    for picture, n in mine:
        run_chunk(picture, n)
    print("done", flush=True)


if __name__ == "__main__":
    main()
