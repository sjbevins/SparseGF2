"""Rebuild or continuously refresh the single-reference campaign status."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from studies.prl_production.single_ref.engine import PointSpec, point_path

PRL_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PRL_ROOT / "manifests" / "runtime"
STATUS_PATH = PRL_ROOT / "STATUS.md"


def _latest_manifest(run_id: str | None) -> Path:
    if run_id:
        path = RUNTIME_ROOT / f"single_ref_{run_id}_manifest.json"
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    paths = list(RUNTIME_ROOT.glob("single_ref_*_manifest.json"))
    if not paths:
        raise FileNotFoundError(f"no single-reference manifests in {RUNTIME_ROOT}")
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


def _atomic_text(path: Path, text: str) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _read_point(
    path: Path,
    cache: dict[Path, tuple[int, int, int, int]],
) -> tuple[int, int, int]:
    if not path.exists():
        return (0, 0, 0)
    stamp = path.stat().st_mtime_ns
    cached = cache.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1:]
    with np.load(path, allow_pickle=False) as data:
        complete = np.asarray(data["complete"], dtype=np.uint8)
        events = np.asarray(data["event_observed"], dtype=np.uint8)
    completed = int(complete.sum())
    observed = int(events[complete == 1].sum())
    full = int(completed == len(complete))
    cache[path] = (stamp, completed, observed, full)
    return completed, observed, full


def refresh_status(
    manifest_path: Path,
    *,
    cache: dict[Path, tuple[int, int, int, int]] | None = None,
) -> str:
    """Scan point checkpoints and atomically refresh STATUS.md."""
    cache = {} if cache is None else cache
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = str(manifest["run_id"])
    state_path = RUNTIME_ROOT / f"single_ref_{run_id}_state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"state": "manifest created", "started_unix": manifest["created_unix"]}
    )
    data_root = Path(manifest["data_root"])
    n_graphs = int(manifest["n_graphs"])
    totals_by_n: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    completed_trajectories = 0
    events = 0
    completed_points = 0

    for raw in manifest["point_order"]:
        point = PointSpec(
            n=int(raw["n"]),
            beta=float(raw["beta"]),
            p=float(raw["p"]),
            n_graphs=int(raw["n_graphs"]),
        )
        completed, observed, full = _read_point(point_path(data_root, point), cache)
        completed_trajectories += completed
        events += observed
        completed_points += full
        totals_by_n[point.n][0] += completed
        totals_by_n[point.n][1] += n_graphs
        totals_by_n[point.n][2] += full

    total_points = int(manifest["n_points"])
    total_trajectories = int(manifest["n_trajectories"])
    started = float(state.get("started_unix", manifest["created_unix"]))
    elapsed = max(0.0, time.time() - started)
    rate = completed_trajectories / elapsed if elapsed > 0 else 0.0
    eta = (total_trajectories - completed_trajectories) / rate if rate > 0 else None
    lines = [
        "# PRL production status",
        "",
        f"**State:** {state.get('state', 'unknown')}",
        f"**Run ID:** {run_id}",
        f"**Profile:** {manifest['profile']}",
        "",
        f"- Points complete: {completed_points:,} / {total_points:,}",
        f"- Trajectories complete: {completed_trajectories:,} / {total_trajectories:,}",
        f"- Observed purification events: {events:,}",
        f"- Elapsed: {elapsed / 3600:.2f} h",
        f"- ETA: {'pending rate estimate' if eta is None else f'{eta / 3600:.2f} h'}",
        "",
        "| n | trajectories | completed points |",
        "|---:|---:|---:|",
    ]
    for n in sorted(totals_by_n):
        completed, total, cells = totals_by_n[n]
        n_points = sum(1 for raw in manifest["point_order"] if int(raw["n"]) == n)
        lines.append(f"| {n} | {completed:,} / {total:,} | {cells:,} / {n_points:,} |")
    lines.extend(["", f"Updated Unix time: {time.time():.3f}", ""])
    _atomic_text(STATUS_PATH, "\n".join(lines))
    return str(state.get("state", "unknown"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.interval < 5:
        parser.error("--interval must be at least 5 seconds")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = _latest_manifest(args.run_id)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    cache: dict[Path, tuple[int, int, int, int]] = {}
    while True:
        state = refresh_status(manifest, cache=cache)
        print(f"status refreshed: {STATUS_PATH} ({state})", flush=True)
        if not args.watch or state in {"complete", "failed", "interrupted"}:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
