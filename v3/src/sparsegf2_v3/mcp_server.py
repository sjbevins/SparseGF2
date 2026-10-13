"""An MCP (Model Context Protocol) server exposing SparseGF2 to LLM agents.

Run it with ``sparsegf2-mcp`` (or ``sparsegf2 mcp``) and register it with a
client, e.g. ``claude mcp add sparsegf2 -- sparsegf2-mcp``. Every tool is a thin
wrapper over :mod:`sparsegf2_v3.api`, so the tools, the CLI and Python callers
share one implementation and one set of error messages.

The ``mcp`` package (``pip install 'sparsegf2-v3[mcp]'``) is imported only when
the server is built, so the simulator itself never depends on it.
"""

import json
from collections.abc import Callable, Sequence
from typing import Annotated, Any

from sparsegf2_v3 import __version__, api

__all__ = ["SERVER_NAME", "build_server", "main", "require_mcp"]

SERVER_NAME = "sparsegf2"

INSTRUCTIONS = """\
SparseGF2 simulates stabilizer (Clifford) circuits fast, especially monitored random
circuits (measurement-induced phase transitions, purification, dynamically generated
codes, erasure protection). Typical flow: capabilities -> inspect_circuit/run_circuit for
single circuits (Stim syntax), or run_ensemble for many random trajectories; use
estimate first and start_ensemble_job + job_status + load_results for long runs. Outputs
are compact summaries; pass output_dir to save full arrays. Read the resource
docs://sparsegf2/quickstart for conventions (seeds, qubit layout, censoring)."""

_INSTALL_HINT = (
    "The SparseGF2 MCP server needs the optional 'mcp' extra: pip install 'sparsegf2-v3[mcp]'"
)


def require_mcp() -> None:
    """Raise ImportError with an install hint if the MCP SDK (v2, ``MCPServer``) is missing."""
    try:
        from mcp.server.mcpserver import MCPServer  # noqa: F401
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


# ---------------------------------------------------------------------------
# tool descriptions (each under 2000 characters, with a worked example)
# ---------------------------------------------------------------------------

_CAPABILITIES = """\
List what the simulator supports: circuit instructions (Stim names plus CLIFFORD_1Q/2Q),
observables for run_circuit, probes and spec keys for run_ensemble, graph families,
limits and examples. Call this first. If the overview is too long, large sections are
replaced by their names; pass section to get one in full.

Example: {"section": "probes"}"""

_INSPECT = """\
Parse a circuit written in Stim's text format without running it. Returns valid, qubit /
measurement / detector / tick counts, an instruction histogram (REPEAT unrolled) and the
normalised text; for invalid text, errors lists every bad line as {"line", "message"}.
Use it to check a circuit before run_circuit.

Example: {"circuit": "H 0\\nCX 0 1\\nREPEAT 10 {\\n  M 0\\n}"}"""

_RUN_CIRCUIT = """\
Run a Stim-language circuit from |0...0>. shots=1 returns the measurement record as a
bitstring (if <= 256 results) plus which outcomes were random; shots>1 returns
per-measurement means, detector and observable flip rates. observables are evaluated on
the final state of the shots=1 run; kinds: entropy, mutual_information,
tripartite_information, entropy_profile, code_dimension, lost_logical_information,
erasure_is_correctable, recovery_probability, random_erasure_statistics, erasure_profile,
contiguous_code_distance, stabilizer_weights, stabilizer_length_distribution, stabilizers,
expectation (needs phases), is_deterministic. Qubit lists are [0,1,2] or "0:3".
observe_ticks=true also evaluates them after every TICK (dynamics vs depth);
tick_stride=k evaluates only every k-th tick.
phases=false is faster but has no outcomes. CLIFFORD_2Q(i) a b applies the two-qubit
Clifford with index i in [0, 11520). output_dir saves full arrays to files.

Example: {"circuit": "H 0\\nCX 0 1\\nTICK\\nM 0", "shots": 1, "seed": 1,
 "observables": [{"kind": "entropy", "region": [1]},
                 {"kind": "expectation", "pauli": "Z0*Z1"}], "observe_ticks": true}"""

_RUN_ENSEMBLE = """\
Run many trajectories of a random monitored circuit (MonitoredCircuit) and return probe
means and standard errors per checkpoint (<= 50 shown), purification statistics (purified
count, censored_fraction, tau median/quantiles among purified, survival curve) and
throughput. Refuses runs estimated above max_seconds (default 60): use start_ensemble_job.
spec keys: <SPEC_KEYS>. graph is {"family": <GRAPHS>, params..., "seed" for random
graphs} (or give edges); graph "realizations": G gives G graphs with circuits_per_graph
circuits each (trajectory t on graph t // circuits_per_graph; omit trajectories to run
all). schedule matching|random_edges|all_edges; reference none|purification|single;
checkpoints "final"|"every_layer"|stride|list; probes [{"kind": <PROBES>, params...,
optional "name", optional own "checkpoints"}]. A layer = gates, then each system qubit
measured with probability measurement_rate. tau=-1 means censored. Vector probes give a
curve over size m (erasure probes: P_C(m), the uncorrectable fraction of m-qubit
erasures) at the last checkpoint plus a coarse depth x size grid (vs_depth).

Example: {"spec": {"num_system": 32, "graph": {"family": "watts_strogatz", "k": 4,
 "beta": 0.1, "seed": 1}, "layers": 256, "measurement_rate": 0.1, "reference": "single",
 "checkpoints": "every_layer", "probes": [{"kind": "code_dimension"}],
 "stop_when_purified": true}, "trajectories": 500, "seed": 7}"""

_ESTIMATE = """\
Rough, deliberately pessimistic runtime estimate (seconds) for run_ensemble with this spec
and trajectory count, whether it fits the synchronous limit, and the result's memory
(result_megabytes). Same arguments as run_ensemble.

Example: {"spec": {"num_system": 256, "graph": {"family": "cycle"}, "layers": 2048,
 "measurement_rate": 0.1, "reference": "single"}, "trajectories": 10000}"""

_START_JOB = """\
Start a long ensemble as a background job; returns immediately with a job_id. The run is
split into chunks (chunk_size trajectories, default about 10 s each), each saved under
output_dir as soon as it finishes, then merged into output_dir/result.npz. Results equal
one run_ensemble call with the same spec and seed. To resume after a cancel or restart,
or to extend a finished run, call again with the same spec and output_dir (seed may be
omitted). Poll with job_status, then call load_results(output_dir).

Example: {"spec": {"num_system": 128, "graph": {"family": "cycle"}, "layers": 1024,
 "measurement_rate": 0.12, "reference": "single", "checkpoints": 16,
 "probes": [{"kind": "code_dimension"}], "stop_when_purified": true},
 "trajectories": 100000, "output_dir": "runs/n128_p012", "seed": 3}"""

_JOB_STATUS = """\
Progress of a background job: status (queued|running|completed|cancelled|failed),
trajectories_done/total, progress, trajectories_per_second, eta_seconds, error. With
wait_seconds (0-60) it blocks until the job ends or the time passes.

Example: {"job_id": "3f2a9c1b7d4e", "wait_seconds": 30}"""

_CANCEL_JOB = """\
Stop a background job after its current chunk. Saved chunks are kept; load_results reads
them, and start_ensemble_job with the same output_dir resumes.

Example: {"job_id": "3f2a9c1b7d4e"}"""

_LIST_JOBS = """\
List the background jobs started by this server process (id, status, progress,
output_dir), newest first.

Example: {}"""

_LOAD_RESULTS = """\
Summarize a saved ensemble: an output directory (including partial chunks of a running or
cancelled job) or a result .npz file. Same summary format as run_ensemble, plus the stored
spec.

Example: {"path": "runs/n128_p012"}"""


def _purification_prompt(num_system: int, graph_family: str, rates: str, trajectories: int) -> str:
    return f"""\
Study the purification transition of monitored random Clifford circuits with SparseGF2.

Setup: {num_system} system qubits on a {graph_family} graph, one reference qubit
Bell-paired with a random system qubit (reference="single", reference_partner="random"),
stop_when_purified=true, checkpoints="every_layer", probes=[{{"kind": "code_dimension"}}],
layers = 8 * num_system. Measurement rates: {rates}. Trajectories per point:
{trajectories}.

Steps:
1. Read docs://sparsegf2/quickstart and call capabilities once.
2. For one rate, call estimate_ensemble; if it is above 60 s use start_ensemble_job with a
   separate output_dir per rate (e.g. runs/purification/n{num_system}_p<rate>) and poll
   job_status with wait_seconds=30; otherwise call run_ensemble with output_dir.
3. For each rate report: censored_fraction, tau_median_purified and quantiles, and the
   survival curve (fraction_unpurified vs layer). Never average tau including -1
   (censored) values.
4. Interpret: in the mixed phase (small rate) the reference stays entangled for a time
   exponential in n; in the pure phase it purifies in O(log n) layers. Repeat at a second
   system size if time allows: a crossing of survival at fixed layer/n locates p_c
   (~0.16 for brickwork Clifford circuits on a ring).
Use fixed integer seeds and keep graph seeds separate from circuit seeds."""


def _erasure_prompt(num_system: int, measurement_rate: float, trajectories: int) -> str:
    return f"""\
Study the erasure protection of the code generated by monitored random Clifford dynamics.

Setup: {num_system} system qubits on a cycle, reference="purification" (every system
qubit entangled with its own reference, so the code starts with {num_system} logical
qubits), measurement_rate={measurement_rate}, layers = 4 * num_system, checkpoints=16,
probes: code_dimension, erasure_first_failure (samples 100) and
uncorrectable_region_counts (max_size 2). Trajectories: {trajectories}.

Steps:
1. Call estimate_ensemble, then run_ensemble (or start_ensemble_job if too slow) with an
   output_dir.
2. Report how the code dimension k(t) decays with depth and, at the final checkpoint,
   P_C(m), the fraction of m-qubit erasures that are uncorrectable (from the first-failure
   samples, and exactly for m <= 2 from the counts). Erasing each qubit independently with
   probability e fails with probability Phi(e) = sum_m C(n,m) e^m (1-e)^(n-m) P_C(m); in
   Python, sparsegf2_v3.analysis.erasure_failure_probability and erasure_tolerance compute
   Phi(e) and e_delta from the saved result.
Report means with standard errors and keep seeds fixed."""


def build_server() -> Any:
    """Construct the :class:`mcp.server.mcpserver.MCPServer` with all tools, resources and prompts."""
    require_mcp()
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import CallToolResult, TextContent, ToolAnnotations

    # Tools return compact JSON text plus the same dict as structured content;
    # the SDK's default conversion would pretty-print it (about 30% longer).
    Result = Annotated[CallToolResult, dict[str, Any]]

    server = MCPServer(
        SERVER_NAME,
        title="SparseGF2 stabilizer simulator",
        instructions=INSTRUCTIONS,
        version=__version__,
    )
    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)
    computes = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)

    def call(fn: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> CallToolResult:
        """Run an api function; anticipated failures become tool errors the model can read."""
        try:
            out = fn(*args, **kwargs)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        text = json.dumps(out, separators=(",", ":"))
        return CallToolResult(content=[TextContent(type="text", text=text)], structured_content=out)

    @server.tool(description=_CAPABILITIES, annotations=read_only)
    def capabilities(section: str | None = None) -> Result:
        return call(api.capabilities, section)

    @server.tool(description=_INSPECT, annotations=read_only)
    def inspect_circuit(circuit: str) -> Result:
        return call(api.inspect_circuit, circuit)

    @server.tool(description=_RUN_CIRCUIT, annotations=computes)
    def run_circuit(
        circuit: str,
        shots: int = 1,
        phases: bool = True,
        seed: int | None = None,
        observables: list[dict[str, Any]] | None = None,
        observe_ticks: bool = False,
        tick_stride: int | None = None,
        num_qubits: int | None = None,
        output_dir: str | None = None,
    ) -> Result:
        return call(
            api.run_circuit,
            circuit,
            shots=shots,
            phases=phases,
            seed=seed,
            observables=observables,
            observe_ticks=observe_ticks,
            tick_stride=tick_stride,
            num_qubits=num_qubits,
            output_dir=output_dir,
            command="mcp:run_circuit",
        )

    # Lists generated from the library, so the description never goes stale.
    run_ensemble_description = (
        _RUN_ENSEMBLE.replace(
            "<SPEC_KEYS>", ", ".join(api.capabilities("ensemble_spec")["ensemble_spec"])
        )
        .replace("<GRAPHS>", "|".join(api.capabilities("graph_families")["graph_families"]))
        .replace("<PROBES>", "|".join(api.capabilities("probes")["probes"]))
    )

    @server.tool(description=run_ensemble_description, annotations=computes)
    def run_ensemble(
        spec: dict[str, Any],
        trajectories: int | None = None,
        seed: int | None = None,
        output_dir: str | None = None,
        max_seconds: float = api.SYNC_ENSEMBLE_MAX_SECONDS,
    ) -> Result:
        if not max_seconds > 0:
            raise ToolError(f"max_seconds must be positive, got {max_seconds:g}")
        # One call: the spec (and its graph bank) is built once, for the
        # estimate and for the run.
        return call(
            api.run_ensemble,
            spec,
            trajectories,
            seed=seed,
            output_dir=output_dir,
            command="mcp:run_ensemble",
            max_seconds=max_seconds,
        )

    @server.tool(description=_ESTIMATE, annotations=read_only)
    def estimate_ensemble(spec: dict[str, Any], trajectories: int | None = None) -> Result:
        return call(api.estimate_ensemble, spec, trajectories)

    @server.tool(description=_START_JOB, annotations=computes)
    def start_ensemble_job(
        spec: dict[str, Any],
        output_dir: str,
        trajectories: int | None = None,
        seed: int | None = None,
        chunk_size: int | None = None,
    ) -> Result:
        return call(
            api.start_ensemble_job,
            spec,
            trajectories,
            output_dir=output_dir,
            seed=seed,
            chunk_size=chunk_size,
            command="mcp:start_ensemble_job",
        )

    @server.tool(description=_JOB_STATUS, annotations=read_only)
    def job_status(job_id: str, wait_seconds: float = 0.0) -> Result:
        return call(api.job_status, job_id, wait_seconds)

    @server.tool(description=_CANCEL_JOB, annotations=computes)
    def cancel_job(job_id: str) -> Result:
        return call(api.cancel_job, job_id)

    @server.tool(description=_LIST_JOBS, annotations=read_only)
    def list_jobs() -> Result:
        return call(api.list_jobs)

    @server.tool(description=_LOAD_RESULTS, annotations=read_only)
    def load_results(path: str) -> Result:
        return call(api.load_results, path)

    @server.resource(
        "docs://sparsegf2/quickstart",
        name="quickstart",
        title="SparseGF2 quickstart for agents",
        description="Concise usage guide: entry points, conventions, pitfalls.",
        mime_type="text/markdown",
    )
    def quickstart() -> str:
        return api.quickstart()

    @server.resource(
        "docs://sparsegf2/capabilities",
        name="capabilities",
        title="SparseGF2 capabilities",
        description="The capabilities() overview as JSON.",
        mime_type="application/json",
    )
    def capabilities_resource() -> str:
        return json.dumps(api.capabilities(), separators=(",", ":"))

    @server.prompt(
        title="Study the purification transition",
        description="Plan and run a measurement-rate sweep of the purification time.",
    )
    def study_purification_transition(
        num_system: int = 64,
        graph_family: str = "cycle",
        rates: str = "0.05, 0.10, 0.15, 0.20, 0.25, 0.30",
        trajectories: int = 2000,
    ) -> str:
        return _purification_prompt(num_system, graph_family, rates, trajectories)

    @server.prompt(
        title="Study erasure protection",
        description="Measure how well the dynamically generated code survives erasures.",
    )
    def study_erasure_protection(
        num_system: int = 32, measurement_rate: float = 0.1, trajectories: int = 500
    ) -> str:
        return _erasure_prompt(num_system, measurement_rate, trajectories)

    return server


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``sparsegf2-mcp``: serve over stdio until the client disconnects."""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(
            "usage: sparsegf2-mcp\n\nServe SparseGF2 over MCP (stdio). Register it with:\n"
            "  claude mcp add sparsegf2 -- sparsegf2-mcp",
            file=sys.stderr,
        )
        return 0
    try:
        server = build_server()
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    server.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
