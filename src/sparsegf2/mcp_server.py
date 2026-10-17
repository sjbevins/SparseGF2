"""An MCP (Model Context Protocol) server exposing SparseGF2 to LLM agents.

Run it with ``sparsegf2-mcp`` (or ``sparsegf2 mcp``) and register it with a
client, e.g. ``claude mcp add sparsegf2 -- sparsegf2-mcp``. Every tool is a thin
wrapper over :mod:`sparsegf2.api`, so the tools, the CLI and Python callers
share one implementation and one set of error messages.

The ``mcp`` package (``pip install 'sparsegf2[mcp]'``) is imported only when
the server is built, so the simulator itself never depends on it.
"""

import json
import logging
from collections.abc import Callable, Sequence
from typing import Annotated, Any

from sparsegf2 import __version__, api

__all__ = ["SERVER_NAME", "build_server", "main", "require_mcp"]

SERVER_NAME = "sparsegf2"

_logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
SparseGF2 simulates stabilizer (Clifford) circuits fast, especially monitored random
circuits (measurement-induced phase transitions, purification, dynamically generated
codes, erasure protection). Typical flow: capabilities -> inspect_circuit/run_circuit for
single circuits (Stim syntax), or run_ensemble for many random trajectories; use
estimate first and start_ensemble_job + job_status + load_results for long runs. Outputs
are compact summaries; pass output_dir to save full arrays. For geometry-correlated erasures,
check the sampler on the graph with sample_erasure_regions before a run and read Phi(e) and
the tolerance from a saved run with erasure_curves. Read the resource
docs://sparsegf2/quickstart for conventions (seeds, qubit layout, censoring)."""

_INSTALL_HINT = (
    "The SparseGF2 MCP server needs the optional 'mcp' extra: pip install 'sparsegf2[mcp]'"
)


def require_mcp() -> None:
    """Raise ImportError with an install hint if the MCP SDK (v2, ``MCPServer``) is missing."""
    try:
        from mcp.server.mcpserver import MCPServer  # noqa: F401
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def _failure_text(exc: Exception) -> str:
    """The tool error for an exception other than the API's ``ValueError``: its type and
    message, so the model always reads what failed (the SDK would send only "Error
    executing tool X"), and that it is a bug to report. The API raises ``ValueError`` for
    every input error (it turns the library's field-naming ``TypeError`` into one), so
    whatever else reaches the server, a ``TypeError`` included, is a bug."""
    return f"{type(exc).__name__}: {exc} (internal error: please report it)"


def _whole_number(value: Any) -> Any:
    """A whole-number float as its int, as the API takes it (JSON Schema's "integer"
    includes 4.0); any other value unchanged, for the strict check that follows."""
    return int(value) if isinstance(value, float) and value.is_integer() else value


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
normalized text; for invalid text, errors lists every bad line as {"line", "message"}.
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
Run random monitored-circuit trajectories: probe means and standard errors per
checkpoint (<= 50 shown), purification statistics (censored_fraction, tau quantiles,
survival) and throughput. num_threads: threads to run on (default <THREADS>, all this
process may use; more run on <THREADS>). Runs estimated above max_seconds (default 60) on
them are refused: use start_ensemble_job.
spec keys: <SPEC_KEYS>. graph: {"family": <GRAPHS>, params..., "seed" if random} (or
give edges); "realizations": G gives G graphs, circuits_per_graph circuits each (omit
trajectories to run all). schedule matching|random_edges|all_edges;
reference none|purification|single; checkpoints "final"|"every_layer"|stride|list;
probes [{"kind": <PROBES>, params..., "name"?, own "checkpoints"?}]. A layer: gates, then
Z measurements at rate measurement_rate. tau=-1 means censored. Vector probes: a curve
over m and a depth x m grid (vs_depth). P_C(m) is over trajectories with a code
(count_with_code), for the counts only if code_dimension or erasure_first_failure covers
their layers (else flagged).

Example: {"spec": {"num_system": 32, "graph": {"family": "watts_strogatz", "k": 4,
 "beta": 0.1, "seed": 1}, "layers": 256, "measurement_rate": 0.1, "reference": "single",
 "checkpoints": "every_layer", "probes": [{"kind": "code_dimension"}],
 "stop_when_purified": true}, "trajectories": 500, "seed": 7, "num_threads": 8}"""

_ESTIMATE = """\
Rough, deliberately pessimistic runtime estimate (seconds) for run_ensemble with this spec
and trajectory count, whether it fits the synchronous limit, and the result's memory
(result_megabytes). threads (a list such as [1, 32]) adds by_threads, the estimate for each
count. layout lists the arrays and files that run_ensemble(output_dir=...) saves, with
shapes, dtypes and uncompressed megabytes (disk use is lower; benchmark_ensemble measures
it). Otherwise the arguments of run_ensemble.

Example: {"spec": {"num_system": 256, "graph": {"family": "cycle"}, "layers": 2048,
 "measurement_rate": 0.1, "reference": "single"}, "trajectories": 10000, "threads": [1, 32]}"""

_BENCHMARK = """\
Measure what a run costs on this machine, before running it. After a warm-up (one trajectory,
which compiles the kernels) it times the first trajectories of the run on one thread and on as
many as this process has, doubling the count until a run takes seconds / 4. It returns
measured timings; projected, the seconds of the whole run for each count in threads (counts
beyond the machine's threads are projected with the simulator's thread model and say so);
estimate_ratio, measured over estimated time per trajectory; and storage: disk_megabytes of
result.npz (compressed, measured on a sample) and memory_megabytes. It writes nothing and takes
up to about 2 x seconds plus the warm-up (seconds at most 30 here). Same spec and
trajectories as run_ensemble; threads defaults to [1, all].

Example: {"spec": {"num_system": 64, "graph": {"family": "cycle"}, "layers": 512,
 "measurement_rate": 0.1, "reference": "single", "probes": [{"kind": "code_dimension"}]},
 "trajectories": 100000, "threads": [1, 32], "seconds": 10}"""

_START_JOB = """\
Start a long ensemble as a background job; returns immediately with a job_id. The run is
split into chunks (chunk_size trajectories, default about 10 s each), each saved under
output_dir as soon as it finishes, then merged into output_dir/result.npz. Results equal
one run_ensemble call with the same spec and seed. To resume after a cancel or restart,
call again with the same spec and output_dir; seed and trajectories may be omitted (the
stored seed and target are used, also for a graph collection). Only a larger trajectories
extends a run. A spec that builds another protocol, or a changed graph file, is refused
before anything runs, and so is a directory another job or process is writing to (one
writer per output_dir). num_threads: threads to run on (default <THREADS>, all this
process may use; more run on <THREADS>). Poll with job_status, then load_results(output_dir).

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
canceled job) or a result .npz file. Same summary format as run_ensemble, plus the stored
spec, exactly as saved: pass it to start_ensemble_job to resume or extend the run, and
storage, the disk use in megabytes of each file and each array (compressed). A "merge
pending" left by a finished job is completed here.

Example: {"path": "runs/n128_p012"}"""


_SAMPLE_REGIONS = """\
Sample geometry-correlated erasure regions on a graph (no circuit) and check how well they mix.
A region A of `size` nodes is drawn with probability proportional to exp(-J b(A)), b(A) the
graph edges leaving A (J > 0 favors compact regions, J = 0 is uniform): the sampler of the
correlated_erasure probe, to try before a long run. graph: {"family": "watts_strogatz", "k": 4,
"beta": 0.1, "seed": 1} with num_nodes (beside it or inside), {"family": "cycle"|"path"},
{"family": "file", "path": ...} or {"edges": [[0, 1], ...], "num_nodes": n}. `chains` runs
(chain i seeded [seed, i]): exact independent samples on a cycle or path (tau_int 1, no R-hat),
else a swap chain with replica exchange. Per coupling: mean_boundary, stderr = sqrt(variance /
ess_boundary); acceptance, exchange_acceptance; tau_int, ess (the slowest of the boundary and a
few nodes' membership); r_hat and its r_hat_limit (what stationary draws of that
autocorrelation allow, shared over the series tested); min_boundary_seen; exact mean/variance
(ring or path closed form, enumeration, or the uniform law at J = 0: exact_source) and
deviation_se. nonincreasing_in_J: no step to the next coupling rises beyond z_critical (1%,
Bonferroni; a test of the estimates, not of Proposition 2); proposition2 gives the worst and
failed steps. mixing: one note per problem with what to change. No note is no proof: repeat
with another seed and a longer burn_in. At most 12 couplings, each |J| <= 1e6; refused if
estimated above 60 s.

Example: {"graph": {"family": "watts_strogatz", "k": 4, "beta": 0.1, "seed": 1},
 "num_nodes": 64, "size": 16, "couplings": [0, 0.5, 1, 2], "samples": 200, "seed": 7}"""

_ERASURE_CURVES = """\
Failure probability and tolerance of the erasure channels of a saved run (output directory or
result.npz). For its erasure_first_failure probe (independent erasure; exact for m <= w where an
uncorrectable_region_counts probe was evaluated at the same layer) and its correlated_erasure
probes (one curve per coupling J), at `layer` (default: each probe's last evaluation), over the
trajectories that hold a code (k > 0): q_plus and count_with_code; phi, Phi(e) = sum_m C(n,m)
e^m (1-e)^(n-m) P(m), mean and stderr at each e (default eight values from 1/n to 1/2): the
uniform rows and J = 0 erase each qubit independently with probability e; a row with J != 0
erases Bin(n, e) qubits as one region drawn from Q_J. e_delta, the tolerance sup{e : Phi(e) <=
delta}: of_mean_phi is the headline, the tolerance of the ensemble-mean Phi, with
of_mean_phi_range for a one-standard-error shift and of_mean_phi_monotone (false: the mean
curve steps down in m, so e_delta is its last crossing); per_code_* describe the spread across
codes (biased upward). For correlated curves, interpolated_sizes are the sizes the probe did not
sample, interpolated_mass the binomial probability they carry at each e, complete whether that
is under 0.1% at those e; e_delta_complete is false when they carry more at e_delta itself
(interpolated_mass_at_e_delta). Needs an erasure_first_failure or correlated_erasure probe. Work
is bounded (a long run uses evenly spaced codes, with a note); at most 12 couplings are shown
(truncated). Writes nothing.

Example: {"path": "runs/erasure", "e": [0.02, 0.05, 0.1], "delta": 0.01, "layer": 128}"""


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
   samples, and exactly for m <= 2 from the counts), averaged over the trajectories that
   still hold a code (count_with_code says how many). Erasing each qubit independently with
   probability e fails with probability Phi(e) = sum_m C(n,m) e^m (1-e)^(n-m) P_C(m); in
   Python, sparsegf2.analysis.erasure_failure_probability and erasure_tolerance compute
   Phi(e) and e_delta from the saved result.
Report means with standard errors and keep seeds fixed."""


def build_server() -> Any:
    """Construct the :class:`mcp.server.mcpserver.MCPServer` with all tools, resources and prompts."""
    require_mcp()
    import numba
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import CallToolResult, TextContent, ToolAnnotations
    from pydantic import BeforeValidator, Strict, StrictBool, StrictFloat

    # Tools return compact JSON text plus the same dict as structured content;
    # the SDK's default conversion would pretty-print it (about 30% longer).
    Result = Annotated[CallToolResult, dict[str, Any]]
    # Arguments are checked as strictly as the API checks them: pydantic's lax default
    # turned true into 1 and "3" into 3 (trajectories: true ran one trajectory). An
    # integer is a JSON integer or a whole-number float, a boolean is true or false, and
    # a number is never a string or a boolean. The input schema is unchanged.
    Int = Annotated[int, Strict(), BeforeValidator(_whole_number)]
    # The threads this process may use, read now so the schemas and descriptions show the
    # real number (16 in Clank's sandbox, whose containers set NUMBA_NUM_THREADS=16 for their
    # 16 CPUs; every thread elsewhere). A larger num_threads runs on this many.
    all_threads = int(numba.config.NUMBA_NUM_THREADS)

    def threads_to_use(num_threads: int) -> int:
        if num_threads < 1:
            raise ToolError(f"num_threads must be at least 1, got {num_threads}")
        return num_threads

    server = MCPServer(
        SERVER_NAME,
        title="SparseGF2 stabilizer simulator",
        instructions=INSTRUCTIONS,
        version=__version__,
    )
    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)
    computes = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)

    def call(fn: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> CallToolResult:
        """Run an api function; every failure becomes a tool error the model can read."""
        try:
            out = fn(*args, **kwargs)
        except ValueError as exc:  # invalid input: the API's message says what to change
            raise ToolError(str(exc)) from exc
        except Exception as exc:  # a bug: the model reads what failed, the log keeps where
            _logger.error("%s failed", getattr(fn, "__name__", "tool"), exc_info=exc)
            raise ToolError(_failure_text(exc)) from exc
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
        shots: Int = 1,
        phases: StrictBool = True,
        seed: Int | None = None,
        observables: list[dict[str, Any]] | None = None,
        observe_ticks: StrictBool = False,
        tick_stride: Int | None = None,
        num_qubits: Int | None = None,
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
        .replace("<THREADS>", str(all_threads))
    )

    @server.tool(description=run_ensemble_description, annotations=computes)
    def run_ensemble(
        spec: dict[str, Any],
        trajectories: Int | None = None,
        seed: Int | None = None,
        output_dir: str | None = None,
        max_seconds: StrictFloat = api.SYNC_ENSEMBLE_MAX_SECONDS,
        num_threads: Int = all_threads,
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
            num_threads=threads_to_use(num_threads),
            command="mcp:run_ensemble",
            max_seconds=max_seconds,
        )

    @server.tool(description=_ESTIMATE, annotations=read_only)
    def estimate_ensemble(
        spec: dict[str, Any],
        trajectories: Int | None = None,
        threads: list[Int] | None = None,
    ) -> Result:
        return call(api.estimate_ensemble, spec, trajectories, threads)

    # A call takes up to about twice `seconds` (plus the warm-up): keep it within the limit
    # for synchronous calls.
    max_benchmark_seconds = api.SYNC_ENSEMBLE_MAX_SECONDS / 2

    @server.tool(description=_BENCHMARK, annotations=read_only)
    def benchmark_ensemble(
        spec: dict[str, Any],
        trajectories: Int | None = None,
        threads: list[Int] | None = None,
        seconds: StrictFloat = 30.0,
        seed: Int = 0,
    ) -> Result:
        if seconds > max_benchmark_seconds:
            raise ToolError(
                f"seconds must be at most {max_benchmark_seconds:g} here (a call takes up to "
                f"twice that, and calls above {api.SYNC_ENSEMBLE_MAX_SECONDS:g} s time out); "
                "use sparsegf2 benchmark in a shell for longer measurements"
            )
        return call(api.benchmark_ensemble, spec, trajectories, threads, seconds, seed)

    @server.tool(
        description=_START_JOB.replace("<THREADS>", str(all_threads)), annotations=computes
    )
    def start_ensemble_job(
        spec: dict[str, Any],
        output_dir: str,
        trajectories: Int | None = None,
        seed: Int | None = None,
        chunk_size: Int | None = None,
        num_threads: Int = all_threads,
    ) -> Result:
        return call(
            api.start_ensemble_job,
            spec,
            trajectories,
            output_dir=output_dir,
            seed=seed,
            chunk_size=chunk_size,
            num_threads=threads_to_use(num_threads),
            command="mcp:start_ensemble_job",
        )

    @server.tool(description=_JOB_STATUS, annotations=read_only)
    def job_status(job_id: str, wait_seconds: StrictFloat = 0.0) -> Result:
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

    @server.tool(description=_SAMPLE_REGIONS, annotations=read_only)
    def sample_erasure_regions(
        graph: dict[str, Any],
        size: Int,
        couplings: StrictFloat | list[StrictFloat],
        samples: Int = 200,
        num_nodes: Int | None = None,
        burn_in: Int | None = None,
        thin: Int = 1,
        tempering: StrictBool = True,
        chains: Int = 4,
        seed: Int | None = None,
        method: str = "auto",
        ladder: str | list[StrictFloat] = "auto",
        max_step: StrictFloat | None = None,
    ) -> Result:
        return call(
            api.sample_erasure_regions,
            graph,
            size,
            couplings,
            samples,
            num_nodes=num_nodes,
            burn_in=burn_in,
            thin=thin,
            tempering=tempering,
            chains=chains,
            seed=seed,
            method=method,
            ladder=ladder,
            max_step=max_step,
        )

    @server.tool(description=_ERASURE_CURVES, annotations=read_only)
    def erasure_curves(
        path: str,
        e: StrictFloat | list[StrictFloat] | None = None,
        delta: StrictFloat = 0.01,
        layer: Int | None = None,
    ) -> Result:
        return call(api.erasure_curves, path, e, delta, layer)

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
