"""
SweepDriver — orchestrates a full ``sizes x p_values x samples`` sweep.

Call path::

    driver = SweepDriver(run_config)
    run_dir = driver.run()

Internally::

    validate_config(run_config)         # raises CompatibilityError if any cell invalid
    RunWriter.begin_run()
    for each (n, p) cell:
        for each sample_seed in [0, n_samples_per_cell):
            SimulationRunner(cell_cfg).run(sample_seed, ...)   # possibly in workers
        RunWriter.write_cell(...)
    RunWriter.finalize()

Parallelism
-----------
The driver uses :class:`concurrent.futures.ProcessPoolExecutor` when
``n_workers > 1``, shipping ``(cfg_dict, seed_start, seed_end, clifford_bytes,
save_tableau, save_realization)`` to each worker. Each worker rebuilds its
CircuitConfig locally and runs one batch of ``batch_size`` samples.

With ``n_workers = 1`` the driver runs inline (useful for tests).
"""
from __future__ import annotations

import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from sparsegf2.circuits.config import CircuitConfig, RunConfig, SampleRecord
from sparsegf2.circuits.runner import SimulationRunner, get_clifford_table
from sparsegf2.circuits.validator import validate_config
from sparsegf2.circuits.writer import RunWriter


# Worker entry point (module-level so it can be pickled)

def _worker_run_batch(args):
    """Run one batch of samples for a single (n, p) cell in a worker process."""
    (cfg_dict, seed_start, seed_end, cliff_bytes,
     save_tableau, save_realization) = args

    # Reconstruct config + Clifford table
    cfg = CircuitConfig(**cfg_dict)
    symp = pickle.loads(cliff_bytes)

    from sparsegf2 import warmup  # ensure JIT hot in worker
    warmup()

    runner = SimulationRunner(cfg, clifford_table=symp, warmup_jit=False)
    records: List[SampleRecord] = []
    for seed in range(seed_start, seed_end):
        rec = runner.run(
            sample_seed=seed,
            save_tableau=save_tableau,
            save_realization=save_realization,
        )
        records.append(rec)
    return records


# Driver

class SweepDriver:
    """Top-level orchestrator for a single sweep."""

    def __init__(
        self,
        run_config: RunConfig,
        *,
        repo_root: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        progress: bool = True,
    ) -> None:
        self.cfg = run_config
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.cache_dir = Path(cache_dir) if cache_dir else (self.repo_root / ".cache")
        self.progress = progress

    # --------------------------------------------------------------

    def run(self) -> Path:
        """Execute the sweep. Returns the run directory."""
        # 1. Pre-flight validation (raises CompatibilityError on failure)
        validate_config(self.cfg)

        # 2. Initialize writer + manifest
        writer = RunWriter(self.cfg, repo_root=self.repo_root)
        writer.begin_run()

        # 3. Prepare Clifford table once
        symp = get_clifford_table(self.cfg.circuit.n_cliffords, cache_dir=self.cache_dir)
        cliff_bytes = pickle.dumps(symp) if self.cfg.n_workers > 1 else None

        # 4. Iterate (n, p) cells
        t0 = time.perf_counter()
        for n in self.cfg.sizes:
            for p in self.cfg.p_values():
                records = self._run_cell(n=int(n), p=float(p), symp=symp,
                                         cliff_bytes=cliff_bytes)
                writer.write_cell(
                    n=int(n), p=float(p), records=records,
                    save_tableaus=self.cfg.save_tableaus,
                    save_realizations=self.cfg.save_realizations,
                )
                if self.progress:
                    self._log_cell(n, p, records, writer)
        wall = time.perf_counter() - t0

        # 5. Finalize
        writer.finalize(wall_seconds=wall)
        return writer.run_dir

    # --------------------------------------------------------------

    def _run_cell(
        self,
        *,
        n: int,
        p: float,
        symp: np.ndarray,
        cliff_bytes: Optional[bytes],
    ) -> List[SampleRecord]:
        cfg = self.cfg.cell_config(n=n, p=p)
        total = self.cfg.n_samples_per_cell

        if self.cfg.n_workers <= 1:
            runner = SimulationRunner(cfg, clifford_table=symp, warmup_jit=False)
            records: List[SampleRecord] = []
            for seed in range(total):
                records.append(runner.run(
                    sample_seed=seed,
                    save_tableau=self.cfg.save_tableaus,
                    save_realization=self.cfg.save_realizations,
                ))
            return records

        # Multi-worker path
        batches: List[Tuple[int, int]] = []
        for start in range(0, total, self.cfg.batch_size):
            end = min(start + self.cfg.batch_size, total)
            batches.append((start, end))
        cfg_dict = asdict(cfg)
        results: List[Tuple[int, List[SampleRecord]]] = []
        with ProcessPoolExecutor(max_workers=self.cfg.n_workers) as pool:
            fut_to_idx = {
                pool.submit(
                    _worker_run_batch,
                    (cfg_dict, s, e, cliff_bytes,
                     self.cfg.save_tableaus, self.cfg.save_realizations),
                ): i
                for i, (s, e) in enumerate(batches)
            }
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                results.append((idx, fut.result()))
        # Re-sort by batch index so seeds appear in order
        results.sort(key=lambda t: t[0])
        ordered: List[SampleRecord] = []
        for _, recs in results:
            ordered.extend(recs)
        return ordered

    # --------------------------------------------------------------

    def _log_cell(
        self,
        n: int,
        p: float,
        records: List[SampleRecord],
        writer: RunWriter,
    ) -> None:
        k_vals = [r.k for r in records]
        abar_vals = [r.final_abar for r in records]
        rt = [r.runtime_total_s for r in records]
        print(
            f"  [n={n:4d} p={p:.4f}] S={len(records):4d}  "
            f"<k>={np.mean(k_vals):.2f}  "
            f"<abar>={np.mean(abar_vals):.2f}  "
            f"<t>={np.mean(rt):.3f}s  -> {writer.cell_dir(n, p).relative_to(writer.run_dir)}",
            flush=True,
        )


__all__ = ["SweepDriver"]
