"""Resume-safe process runner for the exact-layer single-reference campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager, suppress
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

from studies.prl_production.campaign import CampaignProfile, exact_beta, get_profile
from studies.prl_production.single_ref.engine import (
    PointProgress,
    PointSpec,
    prepare_graph_bank,
    run_point,
)

PRL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PRL_ROOT.parents[1]
DEFAULT_DATA_ROOT = PRL_ROOT / "data"
RUNTIME_ROOT = PRL_ROOT / "manifests" / "runtime"
STATUS_PATH = PRL_ROOT / "STATUS.md"
RUNNER_LOCK = RUNTIME_ROOT / "single_ref_runner.lock"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        # Access denied still proves that a process owns this PID.
        return ctypes.get_last_error() == 5
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def _single_runner_lock() -> Iterator[None]:
    """Prevent two campaign runners from checkpointing the same point."""
    RUNNER_LOCK.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    while True:
        try:
            descriptor = os.open(RUNNER_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                owner = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
                owner_pid = int(owner["pid"])
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"cannot validate existing runner lock {RUNNER_LOCK}; inspect it manually"
                ) from exc
            if _pid_exists(owner_pid):
                raise RuntimeError(
                    f"single-reference runner PID {owner_pid} is already active; "
                    "do not launch a second writer"
                ) from None
            # A verified-dead PID means a restart or hard stop left a stale lock.
            RUNNER_LOCK.unlink()
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump({"pid": os.getpid(), "token": token, "created_unix": time.time()}, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        break
    try:
        yield
    finally:
        try:
            owner = json.loads(RUNNER_LOCK.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            owner = {}
        if owner.get("token") == token:
            RUNNER_LOCK.unlink(missing_ok=True)


def _git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip()


def _resolved_profile(args: argparse.Namespace) -> CampaignProfile:
    profile = get_profile(args.profile)
    sizes = tuple(args.n) if args.n else profile.sizes
    if any(n not in profile.sizes for n in sizes):
        raise ValueError(f"--n must select values from {profile.sizes}")
    betas = (
        tuple(exact_beta(value, profile.betas) for value in args.beta)
        if args.beta
        else profile.betas
    )
    n_graphs = args.graphs if args.graphs is not None else profile.n_graphs
    if n_graphs < 1:
        raise ValueError("--graphs must be positive")
    if args.p:
        p_values = tuple(sorted(set(float(value) for value in args.p)))
        if any(not 0.0 <= p <= 1.0 for p in p_values):
            raise ValueError("--p values must lie in [0, 1]")
        p_by_beta = {beta: p_values for beta in betas}
    else:
        p_by_beta = {beta: profile.p_by_beta[beta] for beta in betas}
    return CampaignProfile(
        name=profile.name,
        sizes=sizes,
        betas=betas,
        p_by_beta=p_by_beta,
        n_graphs=n_graphs,
    )


def _beta_priority(beta: float) -> tuple[int, float, float]:
    if beta == 0.0:
        return (0, 0.0, beta)
    return (1, abs(math.log10(beta) + 2.0), beta)


def _points(profile: CampaignProfile) -> list[PointSpec]:
    centers = {
        beta: profile.p_by_beta[beta][len(profile.p_by_beta[beta]) // 2] for beta in profile.betas
    }
    points = [
        PointSpec(n=n, beta=beta, p=p, n_graphs=profile.n_graphs)
        for beta in profile.betas
        for p in profile.p_by_beta[beta]
        for n in profile.sizes
    ]
    points.sort(
        key=lambda point: (
            _beta_priority(point.beta),
            abs(point.p - centers[point.beta]),
            point.n,
            point.p,
        )
    )
    return points


def _source_fingerprint() -> str:
    """Hash every simulator/production source that can affect trajectory bytes."""
    paths = list((REPO_ROOT / "src" / "sparsegf2").rglob("*.py"))
    paths.extend(
        [
            PRL_ROOT / "campaign.py",
            PRL_ROOT / "inputs" / "refinement_centers.csv",
            Path(__file__).resolve().parent / "engine.py",
        ]
    )
    digest = hashlib.sha256()
    digest.update(sys.version.encode())
    for distribution in ("numpy", "numba", "sparsegf2"):
        digest.update(distribution.encode())
        digest.update(metadata.version(distribution).encode())
    for path in sorted(set(paths)):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _run_id(
    profile: CampaignProfile,
    points: list[PointSpec],
    source_fingerprint: str,
) -> str:
    payload = {
        "profile": profile.name,
        "sizes": profile.sizes,
        "betas": profile.betas,
        "n_graphs": profile.n_graphs,
        "points": [(point.n, point.beta_key, point.p_key) for point in points],
        "source_fingerprint": source_fingerprint,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _manifest(
    profile: CampaignProfile,
    points: list[PointSpec],
    args: argparse.Namespace,
    run_id: str,
    source_fingerprint: str,
    data_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "source_fingerprint_sha256": source_fingerprint,
        "created_unix": time.time(),
        "profile": profile.name,
        "sizes": list(profile.sizes),
        "betas": list(profile.betas),
        "p_by_beta": {f"{beta:.9g}": list(profile.p_by_beta[beta]) for beta in profile.betas},
        "n_graphs": profile.n_graphs,
        "n_points": len(points),
        "n_trajectories": len(points) * profile.n_graphs,
        "initial_workers": args.workers,
        "initial_checkpoint_every": args.checkpoint_every,
        "record_traces": args.record_traces,
        "data_root": str(data_root.resolve()),
        "git_head": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_status_porcelain": _git_value("status", "--porcelain=v1", "--untracked-files=no"),
        "python": sys.version,
        "argv": sys.argv,
        "point_order": [asdict(point) for point in points],
    }


def _bounded_parallel[T, R](
    executor: ProcessPoolExecutor,
    items: Iterable[T],
    submit: Callable[[ProcessPoolExecutor, T], Future[R]],
    *,
    max_pending: int,
) -> Iterator[R]:
    iterator = iter(items)
    pending: set[Future[R]] = set()
    for _ in range(max_pending):
        try:
            pending.add(submit(executor, next(iterator)))
        except StopIteration:
            break
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            yield future.result()
            with suppress(StopIteration):
                pending.add(submit(executor, next(iterator)))


def _write_status(
    *,
    state: str,
    profile: CampaignProfile,
    run_id: str,
    completed_points: int,
    completed_trajectories: int,
    events: int,
    total_points: int,
    total_trajectories: int,
    started: float,
    last_point: PointProgress | None = None,
    note: str | None = None,
) -> None:
    elapsed = max(0.0, time.time() - started)
    point_rate = completed_points / elapsed if elapsed > 0 else 0.0
    eta_s = (total_points - completed_points) / point_rate if point_rate > 0 else None
    lines = [
        "# PRL production status",
        "",
        f"**State:** {state}",
        f"**Run ID:** {run_id}",
        f"**Profile:** {profile.name}",
        "",
        f"- Points complete: {completed_points:,} / {total_points:,}",
        f"- Trajectories complete: {completed_trajectories:,} / {total_trajectories:,}",
        f"- Observed purification events: {events:,}",
        f"- Elapsed: {elapsed / 3600:.2f} h",
        f"- ETA: {'pending rate estimate' if eta_s is None else f'{eta_s / 3600:.2f} h'}",
    ]
    if last_point is not None:
        point = last_point.point
        lines.extend(
            [
                "",
                "Last completed point:",
                (
                    f"n={point.n}, beta={point.beta:.9g}, p={point.p:.6f} "
                    f"({last_point.completed}/{point.n_graphs} trajectories)"
                ),
            ]
        )
    if note:
        lines.extend(["", note])
    lines.extend(["", f"Updated Unix time: {time.time():.3f}", ""])
    temp = STATUS_PATH.with_name(f".{STATUS_PATH.name}.{os.getpid()}.tmp")
    try:
        temp.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        os.replace(temp, STATUS_PATH)
    finally:
        if temp.exists():
            temp.unlink()


def _shutdown(executor: ProcessPoolExecutor, *, terminate: bool) -> None:
    if terminate and hasattr(executor, "terminate_workers"):
        executor.terminate_workers()
    else:
        executor.shutdown(wait=not terminate, cancel_futures=terminate)


def _prepare_banks(data_root: Path, points: list[PointSpec], workers: int) -> None:
    unique: dict[tuple[int, int, int], PointSpec] = {}
    for point in points:
        unique[(point.n, point.beta_key, point.n_graphs)] = point
    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        for _ in _bounded_parallel(
            executor,
            unique.values(),
            lambda pool, point: pool.submit(prepare_graph_bank, data_root, point),
            max_pending=max(1, workers * 2),
        ):
            pass
    except BaseException:
        _shutdown(executor, terminate=True)
        raise
    else:
        _shutdown(executor, terminate=False)


def _run_points(
    data_root: Path,
    points: list[PointSpec],
    profile: CampaignProfile,
    args: argparse.Namespace,
    run_id: str,
    state_path: Path,
    started: float,
) -> tuple[int, int, int, PointProgress | None]:
    completed_points = 0
    completed_trajectories = 0
    events = 0
    last_result: PointProgress | None = None
    last_report = 0.0
    executor = ProcessPoolExecutor(max_workers=args.workers)
    try:
        results = _bounded_parallel(
            executor,
            points,
            lambda pool, point: pool.submit(
                run_point,
                data_root,
                point,
                checkpoint_every=args.checkpoint_every,
                record_traces=args.record_traces,
            ),
            max_pending=max(1, args.workers * 2),
        )
        for result in results:
            last_result = result
            if result.is_complete:
                completed_points += 1
            completed_trajectories += result.completed
            events += result.events
            now = time.time()
            if now - last_report >= 20 or completed_points == len(points):
                state = {
                    "run_id": run_id,
                    "state": "running",
                    "completed_points": completed_points,
                    "completed_trajectories": completed_trajectories,
                    "events": events,
                    "total_points": len(points),
                    "total_trajectories": len(points) * profile.n_graphs,
                    "runner_pid": os.getpid(),
                    "started_unix": started,
                    "updated_unix": now,
                    "last_point": asdict(result.point),
                }
                _atomic_json(state_path, state)
                _write_status(
                    state="running",
                    profile=profile,
                    run_id=run_id,
                    completed_points=completed_points,
                    completed_trajectories=completed_trajectories,
                    events=events,
                    total_points=len(points),
                    total_trajectories=len(points) * profile.n_graphs,
                    started=started,
                    last_point=result,
                )
                print(
                    f"[{completed_points:,}/{len(points):,}] "
                    f"n={result.point.n} beta={result.point.beta:.9g} p={result.point.p:.6f} "
                    f"{result.completed}/{result.point.n_graphs}",
                    flush=True,
                )
                last_report = now
    except BaseException:
        _shutdown(executor, terminate=True)
        raise
    else:
        _shutdown(executor, terminate=False)
    return completed_points, completed_trajectories, events, last_result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "pilot", "production"), default="smoke")
    parser.add_argument("--confirm-production", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("SPARSEGF2_WORKERS", "8"))
    )
    parser.add_argument("--n", type=int, nargs="+")
    parser.add_argument("--beta", type=float, nargs="+")
    parser.add_argument("--p", type=float, nargs="+")
    parser.add_argument("--graphs", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--record-traces", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--print-run-id",
        action="store_true",
        help="print the deterministic run ID without writing manifests or data",
    )
    parser.add_argument("--max-points", type=int)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 32:
        parser.error("--workers must lie in [1, 32]")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    if args.max_points is not None and args.max_points < 1:
        parser.error("--max-points must be positive")
    if args.profile == "production" and not args.confirm_production:
        parser.error("the production profile requires --confirm-production")
    return args


def _main_unlocked(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        profile = _resolved_profile(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    points = _points(profile)
    if args.max_points is not None:
        points = points[: args.max_points]
    if args.record_traces and (profile.n_graphs > 32 or len(points) > 20):
        raise SystemExit("--record-traces is restricted to <=32 graphs and <=20 points")

    source_fingerprint = _source_fingerprint()
    run_id = _run_id(profile, points, source_fingerprint)
    if args.print_run_id:
        print(run_id, flush=True)
        return 0
    data_root = Path(args.data_root) / "runs" / run_id
    manifest_path = RUNTIME_ROOT / f"single_ref_{run_id}_manifest.json"
    state_path = RUNTIME_ROOT / f"single_ref_{run_id}_state.json"
    manifest = _manifest(
        profile,
        points,
        args,
        run_id,
        source_fingerprint,
        data_root,
    )
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        stable_keys = (
            "run_id",
            "profile",
            "sizes",
            "betas",
            "p_by_beta",
            "n_graphs",
            "n_points",
            "n_trajectories",
            "data_root",
        )
        if any(existing.get(key) != manifest.get(key) for key in stable_keys):
            raise SystemExit(f"existing manifest conflicts with resolved run: {manifest_path}")
    else:
        _atomic_json(manifest_path, manifest)

    total_trajectories = len(points) * profile.n_graphs
    print(
        f"profile={profile.name} points={len(points):,} "
        f"trajectories={total_trajectories:,} workers={args.workers} run_id={run_id}",
        flush=True,
    )
    if args.dry_run:
        return 0

    started = time.time()
    completed_points = 0
    completed_trajectories = 0
    events = 0
    last_result: PointProgress | None = None
    state = "failed"
    note = "The run ended before completion."
    exit_code = 1
    _atomic_json(
        state_path,
        {
            "run_id": run_id,
            "state": "preparing_graphs",
            "total_points": len(points),
            "total_trajectories": total_trajectories,
            "runner_pid": os.getpid(),
            "workers": args.workers,
            "checkpoint_every": args.checkpoint_every,
            "started_unix": started,
            "updated_unix": started,
        },
    )
    _write_status(
        state="preparing graph banks",
        profile=profile,
        run_id=run_id,
        completed_points=0,
        completed_trajectories=0,
        events=0,
        total_points=len(points),
        total_trajectories=total_trajectories,
        started=started,
    )
    try:
        _prepare_banks(data_root, points, args.workers)
        if args.prepare_only:
            state = "graphs prepared"
        else:
            completed_points, completed_trajectories, events, last_result = _run_points(
                data_root,
                points,
                profile,
                args,
                run_id,
                state_path,
                started,
            )
            state = "complete"
        note = "Atomic data files are complete and resume-safe."
        exit_code = 0
    except KeyboardInterrupt:
        state = "interrupted"
        note = "The run stopped cleanly. Atomic point checkpoints remain resume-safe."
        exit_code = 130
    finally:
        final_state = {
            "run_id": run_id,
            "state": state,
            "completed_points": completed_points,
            "completed_trajectories": completed_trajectories,
            "events": events,
            "total_points": len(points),
            "total_trajectories": total_trajectories,
            "runner_pid": os.getpid(),
            "workers": args.workers,
            "checkpoint_every": args.checkpoint_every,
            "started_unix": started,
            "updated_unix": time.time(),
        }
        _atomic_json(state_path, final_state)
        _write_status(
            state=state,
            profile=profile,
            run_id=run_id,
            completed_points=completed_points,
            completed_trajectories=completed_trajectories,
            events=events,
            total_points=len(points),
            total_trajectories=total_trajectories,
            started=started,
            last_point=last_result,
            note=note,
        )
    return exit_code


def main(argv: list[str] | None = None) -> int:
    with _single_runner_lock():
        return _main_unlocked(argv)


if __name__ == "__main__":
    raise SystemExit(main())
