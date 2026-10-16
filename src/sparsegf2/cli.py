"""The ``sparsegf2`` command line: a thin shell over :mod:`sparsegf2.api`.

Every subcommand prints a readable summary, or with ``--json`` exactly one
JSON object on stdout (logs and progress go to stderr), so scripts and LLM
agents can parse it. Exit codes: 0 success, 2 invalid input or usage (including
``inspect`` of an invalid circuit, a file that cannot be read and an output
directory that cannot be written), 1 internal error, 130 interrupted (Ctrl-C).
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NoReturn

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

_MAIN_HELP = """\
examples:
  sparsegf2 capabilities --json
  echo 'H 0
  CX 0 1
  M 0 1' | sparsegf2 run - --shots 1000 --seed 1
  sparsegf2 inspect circuit.stim
  sparsegf2 ensemble spec.json --trajectories 1000 --seed 7 --output-dir runs/p010
  sparsegf2 benchmark spec.json --trajectories 100000 --threads 1 32
  sparsegf2 results runs/p010
  sparsegf2 mcp                   # MCP server over stdio (for Claude and other agents)

Every command accepts --json: one JSON object on stdout, logs on stderr.
Exit codes: 0 ok, 2 invalid input (also: inspect of an invalid circuit),
1 internal error, 130 interrupted.
"""

_RUN_HELP = """\
examples:
  sparsegf2 run bell.stim --shots 1000 --seed 1
  sparsegf2 run circuit.stim --observables '[{"kind": "entropy", "region": "0:4"}]'
  sparsegf2 run circuit.stim --observables @obs.json --observe-ticks --json
  sparsegf2 run circuit.stim --observables @obs.json --observe-ticks --tick-stride 10
  sparsegf2 run circuit.stim --no-phases     # faster; no outcomes, only random flags
"""

_INSPECT_HELP = """\
examples:
  sparsegf2 inspect circuit.stim
  cat circuit.stim | sparsegf2 inspect - --json

Exits with 2 (after printing the errors) if the circuit is invalid.
"""

_ENSEMBLE_HELP = """\
examples:
  sparsegf2 ensemble spec.json --trajectories 1000 --seed 7
  sparsegf2 ensemble spec.json --trajectories 100000 --output-dir runs/a --chunk-size 5000
  sparsegf2 ensemble runs/a/spec.json       # spec.json from a run: repeats it exactly
                                            # (if made under this randomness contract)

SPEC.json holds MonitoredCircuit parameters, e.g.
  {"num_system": 64, "graph": {"family": "cycle"}, "layers": 256,
   "measurement_rate": 0.1, "reference": "single", "checkpoints": 16,
   "probes": [{"kind": "code_dimension"}], "stop_when_purified": true}
or {"spec": {...}, "trajectories": N, "seed": S} (flags override the file).
With --chunk-size the run is saved chunk by chunk to --output-dir and can be
resumed by running the same command again.
"""

_BENCHMARK_HELP = """\
examples:
  sparsegf2 benchmark spec.json --trajectories 100000 --threads 1 32
  sparsegf2 benchmark spec.json --trajectories 5000 --seconds 10 --json

Times the first trajectories of the run on one thread and on as many as this machine
has, then projects the seconds of the whole run to each count in --threads (counts
beyond the machine's threads are projected with the simulator's thread model), and
measures the size of result.npz. Writes nothing. SPEC.json is read as by ensemble.
"""

_RESULTS_HELP = """\
examples:
  sparsegf2 results runs/a              # an output directory (includes partial chunks)
  sparsegf2 results runs/a/result.npz --json
"""

_CAPABILITIES_HELP = """\
examples:
  sparsegf2 capabilities
  sparsegf2 capabilities --section probes --json
"""

_MCP_HELP = """\
examples:
  sparsegf2 mcp                          # serve over stdio
  claude mcp add sparsegf2 -- sparsegf2 mcp
Needs the optional extra: pip install 'sparsegf2[mcp]'.
"""


class _UsageError(Exception):
    """Raised instead of argparse's sys.exit so main() can report it its own way."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(f"{self.prog}: {message}")


def _build_parser() -> _Parser:
    fmt = argparse.RawDescriptionHelpFormatter
    parser = _Parser(
        prog="sparsegf2",
        description=_MAIN_HELP + "\nSparseGF2: fast stabilizer simulation of monitored circuits.",
        formatter_class=fmt,
    )
    from sparsegf2 import __version__

    parser.add_argument("--version", action="version", version=f"sparsegf2 {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", parser_class=_Parser)
    sub.required = True

    def add(name: str, help_text: str, examples: str) -> _Parser:
        p = sub.add_parser(
            name, help=help_text, description=examples + "\n" + help_text, formatter_class=fmt
        )
        p.add_argument("--json", action="store_true", help="print one JSON object on stdout")
        return p

    p = add(
        "capabilities",
        "List supported instructions, observables, probes and limits.",
        _CAPABILITIES_HELP,
    )
    p.add_argument("--section", help="show one section in full (e.g. probes, observables)")

    p = add(
        "inspect",
        "Parse a Stim-language circuit; report counts and errors with line numbers.",
        _INSPECT_HELP,
    )
    p.add_argument("file", metavar="FILE", help="circuit file, or - for stdin")

    p = add("run", "Run a circuit and summarize measurements and observables.", _RUN_HELP)
    p.add_argument("file", metavar="FILE", help="circuit file, or - for stdin")
    p.add_argument("--shots", type=int, default=1, help="number of shots (default 1)")
    p.add_argument("--seed", type=int, default=None, help="random seed (default: fresh, reported)")
    p.add_argument(
        "--phases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="track signs (default); --no-phases is faster but has no outcomes",
    )
    p.add_argument("--observables", help="JSON list of observables, or @file.json")
    p.add_argument(
        "--observe-ticks", action="store_true", help="also evaluate observables after every TICK"
    )
    p.add_argument(
        "--tick-stride",
        type=int,
        default=None,
        help="with --observe-ticks, evaluate only every k-th TICK (and the last)",
    )
    p.add_argument("--num-qubits", type=int, default=None, help="simulate extra idle qubits")
    p.add_argument("--output-dir", help="save full arrays, the circuit and provenance here")

    p = add(
        "ensemble", "Run an ensemble of random monitored circuits from a JSON spec.", _ENSEMBLE_HELP
    )
    p.add_argument("spec", metavar="SPEC.json", help="spec file, or - for stdin")
    p.add_argument(
        "--trajectories",
        type=int,
        default=None,
        help=(
            "number of trajectories (default for a graph collection: all of its circuits; "
            "for a resumed --chunk-size run: the target stored in its --output-dir)"
        ),
    )
    p.add_argument(
        "--seed", type=int, default=None, help="random seed (default: from file, else fresh)"
    )
    p.add_argument("--output-dir", help="save result.npz, spec.json and provenance.json here")
    p.add_argument(
        "--chunk-size", type=int, default=None, help="run in resumable chunks (needs --output-dir)"
    )
    p.add_argument("--threads", type=int, default=None, help="limit the number of CPU threads")

    p = add(
        "benchmark",
        "Measure the speed and storage of an ensemble on this machine; writes nothing.",
        _BENCHMARK_HELP,
    )
    p.add_argument("spec", metavar="SPEC.json", help="spec file, or - for stdin")
    p.add_argument(
        "--trajectories",
        type=int,
        default=None,
        help="trajectories of the planned run (default for a graph collection: all its circuits)",
    )
    p.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=None,
        help="thread counts to report (default: 1 and every thread Numba can use)",
    )
    p.add_argument(
        "--seconds",
        type=float,
        default=30.0,
        help="time target per measurement; the call takes up to about twice this (default 30)",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="random seed (default: from file, else 0)"
    )

    p = add(
        "results", "Summarize a saved ensemble (result.npz or an output directory).", _RESULTS_HELP
    )
    p.add_argument("path", metavar="PATH", help="result .npz file or output directory")

    add("mcp", "Start the MCP server over stdio.", _MCP_HELP)
    return parser


# ---------------------------------------------------------------------------
# input helpers
# ---------------------------------------------------------------------------


def _read_text(path: str) -> str:
    """The text of a file the user named (``-``: stdin); a file that cannot be read is an
    input error (exit code 2) that names it."""
    if path == "-":
        return sys.stdin.read()
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"no such file: {path!r}")
    try:
        return p.read_text(encoding="utf-8")
    except OSError as exc:  # no permission, or (Windows) held by another program
        raise ValueError(f"could not read {path!r}: {exc}") from None
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path!r} is not UTF-8 text: {exc}") from None


def _read_json(text: str, what: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{what} is not valid JSON: {exc}") from None


def _json_arg(value: str | None, what: str) -> Any:
    if value is None:
        return None
    if value.startswith("@"):
        return _read_json(_read_text(value[1:]), what)
    return _read_json(value, what)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# human-readable output
# ---------------------------------------------------------------------------


def _scalar(value: Any) -> str:
    if isinstance(value, str):
        return value if value and "\n" not in value else json.dumps(value)
    return json.dumps(value)


def _human(obj: Any, indent: int = 0) -> list[str]:
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict) and value:
                lines.append(f"{pad}{key}:")
                lines.extend(_human(value, indent + 1))
            elif isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
                lines.append(f"{pad}{key}:")
                for item in value:
                    sub = _human(item, indent + 2)
                    if sub:
                        lines.append(f"{pad}  - {sub[0].strip()}")
                        lines.extend(sub[1:])
            elif isinstance(value, str) and "\n" in value:
                lines.append(f"{pad}{key}: |")
                lines.extend(f"{pad}  {line}" for line in value.splitlines())
            else:
                lines.append(f"{pad}{key}: {_scalar(value)}")
    elif isinstance(obj, list):
        lines.extend(f"{pad}- {_scalar(v)}" for v in obj)
    else:
        lines.append(f"{pad}{_scalar(obj)}")
    return lines


def _emit(result: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result))
    else:
        print("\n".join(_human(result)))


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


# Each command returns (result, exit code); the result is printed either way.
_Outcome = tuple[dict[str, Any], int]


def _cmd_capabilities(args: argparse.Namespace) -> _Outcome:
    from sparsegf2 import api

    return api.capabilities(args.section), EXIT_OK


def _cmd_inspect(args: argparse.Namespace) -> _Outcome:
    from sparsegf2 import api

    result = api.inspect_circuit(_read_text(args.file))
    return result, EXIT_OK if result["valid"] else EXIT_USAGE


def _cmd_run(args: argparse.Namespace) -> _Outcome:
    from sparsegf2 import api

    result = api.run_circuit(
        _read_text(args.file),
        shots=args.shots,
        phases=args.phases,
        seed=args.seed,
        observables=_json_arg(args.observables, "--observables"),
        observe_ticks=args.observe_ticks,
        tick_stride=args.tick_stride,
        num_qubits=args.num_qubits,
        output_dir=args.output_dir,
        command=_command_line(),
    )
    return result, EXIT_OK


def _read_spec_file(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    """``(spec, trajectories, seed)`` from SPEC.json, the flags overriding the file."""
    data = _read_json(_read_text(args.spec), "SPEC.json")
    if isinstance(data, dict) and "spec" in data:
        # graph_file is the record of a graph file that a run writes beside its spec
        unknown = sorted(set(data) - {"spec", "trajectories", "seed", "graph_file"})
        if unknown:
            raise ValueError(
                f"SPEC.json: unknown top-level key(s) {unknown}; expected spec, trajectories, "
                "seed (and graph_file, which a run writes beside the spec)"
            )
        spec, trajectories, seed = data["spec"], data.get("trajectories"), data.get("seed")
    else:
        spec, trajectories, seed = data, None, None
    trajectories = args.trajectories if args.trajectories is not None else trajectories
    seed = args.seed if args.seed is not None else seed
    return spec, trajectories, seed


def _cmd_benchmark(args: argparse.Namespace) -> _Outcome:
    from sparsegf2 import api

    spec, trajectories, seed = _read_spec_file(args)
    _log("benchmarking (this takes about twice --seconds, plus the first compile)")
    result = api.benchmark_ensemble(
        spec,
        trajectories,
        threads=args.threads,
        seconds=args.seconds,
        seed=0 if seed is None else seed,
    )
    return result, EXIT_OK


def _human_benchmark(result: dict[str, Any]) -> list[str]:
    """A short table of a :func:`sparsegf2.api.benchmark_ensemble` result."""
    storage = result["storage"]
    lines = [
        f"{result['trajectories']} trajectories, seed {result['seed']}; this process has "
        f"{result['threads_available']} threads ({result['cpus']} cpus); warm-up "
        f"{result['warmup_seconds']} s",
        "",
        "measured     threads  trajectories  seconds  trajectories/s",
    ]
    lines += [
        f"             {m['threads']:>7}  {m['trajectories']:>12}  {m['seconds']:>7}  "
        f"{m['trajectories_per_second']:>14}"
        for m in result["measured"]
    ]
    lines += ["", "projected    threads  seconds (whole run)  basis"]
    lines += [
        f"             {row['threads']:>7}  {row['seconds']:>19}  {row['basis']}"
        for row in result["projected"]
    ]
    lines += [
        "",
        f"estimate_ensemble said {result['estimated_seconds']} s; measured / estimated per "
        f"trajectory = {result['estimate_ratio']}",
        f"storage: disk {storage['disk_megabytes']} MB (result.npz), memory "
        f"{storage['memory_megabytes']} MB",
    ]
    lines += [f"note: {note}" for note in result.get("notes", [])]
    return lines


def _cmd_ensemble(args: argparse.Namespace) -> _Outcome:
    from sparsegf2 import api

    spec, trajectories, seed = _read_spec_file(args)
    # Without a count the API decides: a graph collection (graph.realizations, or a graph
    # file holding several graphs) runs all of its circuits, a resumed job runs to its
    # stored target, and a single graph is refused with a message.
    if args.chunk_size is None:
        estimate = api.estimate_ensemble(spec, trajectories)
        _log(
            f"running {estimate['trajectories']} trajectories "
            f"(estimated <= {estimate['estimated_seconds']} s)"
        )
        result = api.run_ensemble(
            spec,
            trajectories,
            seed=seed,
            output_dir=args.output_dir,
            num_threads=args.threads,
            command=_command_line(),
        )
        return result, EXIT_OK
    if args.output_dir is None:
        raise ValueError("--chunk-size needs --output-dir (chunks are saved there)")
    return _run_job_in_foreground(spec, trajectories, seed, args)


def _run_job_in_foreground(
    spec: Any, trajectories: Any, seed: Any, args: argparse.Namespace
) -> _Outcome:
    """Run a chunked, resumable job and show its progress on stderr until it ends.

    On Ctrl-C the job stops after its current chunk, the partial summary is
    printed and the exit code is 130; running the same command resumes.
    """
    from sparsegf2 import api

    status = api.start_ensemble_job(
        spec,
        trajectories,
        output_dir=args.output_dir,
        seed=seed,
        chunk_size=args.chunk_size,
        num_threads=args.threads,
        command=_command_line(),
    )
    job_id = status["job_id"]
    start_notes = list(status.get("notes", []))  # the resume's notes, shown by the start only
    if status["resumed_from"]:
        _log(f"resuming: {status['resumed_from']} trajectories already saved")
    interrupted = False
    try:
        while status["status"] in ("queued", "running"):
            status = api.job_status(job_id, wait_seconds=2.0)
            eta = status.get("eta_seconds")
            _log(
                f"{status['trajectories_done']}/{status['trajectories_total']} trajectories"
                + (f", eta {eta:.0f} s" if eta is not None else "")
            )
    except KeyboardInterrupt:
        interrupted = True
        api.cancel_job(job_id)
        _log("interrupted: finishing the current chunk; rerun the same command to resume")
        status = api.job_status(job_id, wait_seconds=60)
    if status["status"] == "failed":
        raise RuntimeError(status.get("error", "job failed"))
    summary = api.load_results(args.output_dir)
    summary["job"] = {
        k: status[k] for k in ("job_id", "status", "trajectories_done", "resumed_from")
    }
    # the resume's notes and the job's own (a merge left pending, say)
    notes = [*start_notes, *(n for n in status.get("notes", []) if n not in start_notes)]
    if notes:
        summary["job"]["notes"] = notes
    return summary, EXIT_INTERRUPTED if interrupted else EXIT_OK


def _cmd_results(args: argparse.Namespace) -> _Outcome:
    from sparsegf2 import api

    return api.load_results(args.path), EXIT_OK


def _command_line() -> str:
    return shlex.join(["sparsegf2", *sys.argv[1:]])


_COMMANDS = {
    "capabilities": _cmd_capabilities,
    "inspect": _cmd_inspect,
    "run": _cmd_run,
    "ensemble": _cmd_ensemble,
    "benchmark": _cmd_benchmark,
    "results": _cmd_results,
}

# Commands whose result is shown as a table rather than as the generic listing.
_TABLES: dict[str, Callable[[dict[str, Any]], list[str]]] = {"benchmark": _human_benchmark}


def _error(kind: str, message: str, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"error": {"type": kind, "message": message}}))
    _log(f"error: {message}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI with ``argv`` (default ``sys.argv[1:]``); returns the exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in argv
    try:
        args = _build_parser().parse_args(argv)
    except _UsageError as exc:
        _error("UsageError", str(exc), as_json)
        return EXIT_USAGE
    except SystemExit as exc:  # --help and --version
        return int(exc.code or 0)
    if args.command == "mcp":
        from sparsegf2 import mcp_server

        try:
            mcp_server.require_mcp()
        except ImportError as exc:
            _error("UsageError", str(exc), as_json)
            return EXIT_USAGE
        return mcp_server.main([])
    start = time.perf_counter()
    try:
        result, code = _COMMANDS[args.command](args)
    except ValueError as exc:
        _error("ValidationError", str(exc), as_json)
        return EXIT_USAGE
    except KeyboardInterrupt:
        _error("Interrupted", "interrupted", as_json)
        return EXIT_INTERRUPTED
    except Exception as exc:  # report, never dump a traceback on stdout
        _error(type(exc).__name__, f"internal error: {exc}", as_json)
        return EXIT_INTERNAL
    if as_json or args.command not in _TABLES:
        _emit(result, as_json)
    else:
        print("\n".join(_TABLES[args.command](result)))
    if not as_json:
        _log(f"done in {time.perf_counter() - start:.2f} s")
    return code


if __name__ == "__main__":
    sys.exit(main())
