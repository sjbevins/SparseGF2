# Getting started with the SparseGF2 MCP server

This guide sets up the SparseGF2 MCP server so that an AI agent (Claude Code, Claude
Desktop, or any other client that speaks the Model Context Protocol) can run your
simulations for you: inspect and run Stim circuits, run ensembles of monitored random
Clifford circuits, launch long runs in the background, and summarize the saved results.

You will:

1. install the server and check that it starts (section 2),
2. connect it to your client (section 3),
3. talk to it from a short Python script, without any agent, to see that it works
   (section 4),
4. run a first session: a circuit, a purification ensemble, and a small erasure study
   with graph collections, per-probe checkpoints and saved codes, then analyse the saved
   result in Python (section 5).

Sections 6 and 7 cover files, long runs and troubleshooting. Every command, script and
output below was run while writing this guide (SparseGF2 3.0.0.dev9, MCP Python SDK 2.2
and 2.3, Python 3.13, Linux, 4 cores); outputs are real and only shortened where marked
with `...`.

If you have not run a simulation yet, start with [`GETTING_STARTED.md`](GETTING_STARTED.md),
the guide to the simulator itself. [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md) explains the engine,
and [`LLM_INTERFACE.md`](LLM_INTERFACE.md) is the complete reference for the JSON
interface that the server exposes.

---

## 1. What the server is

`sparsegf2-mcp` is a small program that speaks MCP over standard input and output
(stdio). Your client starts it as a child process and offers its tools to the model. Every
tool is a thin wrapper over the functions in
[`src/sparsegf2/api.py`](../src/sparsegf2/api.py), the same functions the
`sparsegf2` command line and Python's `sparsegf2.api` use, so arguments, results and
error messages are identical everywhere.

### Tools

| Tool | What it does | Key arguments |
|---|---|---|
| `capabilities` | Lists everything the simulator supports: instructions, observables, probes, spec keys, graph families, limits, examples. The agent calls it first. | `section` (e.g. `"probes"`) for one section in full |
| `inspect_circuit` | Parses a Stim-format circuit without running it: counts, instruction histogram, or every bad line with its line number. | `circuit` |
| `run_circuit` | Runs a Stim-format circuit from \|0...0>. | `circuit`, `shots`, `seed`, `observables`, `observe_ticks`, `tick_stride`, `phases`, `output_dir` |
| `run_ensemble` | Runs many trajectories of a random monitored circuit (a `MonitoredCircuit` spec) and returns probe means and standard errors per checkpoint, purification statistics and throughput. Refuses runs estimated above `max_seconds`. | `spec`, `trajectories` (optional for a graph collection), `seed`, `output_dir`, `max_seconds` (default 60) |
| `estimate_ensemble` | A deliberately pessimistic runtime estimate, the trajectory count and the result's memory, before anything runs. | `spec`, `trajectories` |
| `start_ensemble_job` | Starts a long ensemble as a background job, saved chunk by chunk; resumes or extends a run in the same directory. Returns at once with a `job_id`. | `spec`, `output_dir` (required), `trajectories`, `seed`, `chunk_size` |
| `job_status` | Progress of a job: status, trajectories done, throughput, ETA, error. | `job_id`, `wait_seconds` (0 to 60: block until done or timeout) |
| `cancel_job` | Stops a job after its current chunk; saved chunks are kept. | `job_id` |
| `list_jobs` | The jobs started by this server process. | none |
| `load_results` | Summarizes a saved run (an output directory, including the chunks of an unfinished job, or a `.npz` file), in the same format as `run_ensemble`, plus the stored spec. | `path` |

Each tool's description, which the model reads, contains a worked example of its
arguments. The lists of spec keys, probe kinds and graph families inside the
`run_ensemble` description are generated from the library, so they cannot go stale.

### Prompts and resources

* Prompts (ready-made study plans; how a client shows them varies, see section 5.5):
  * `study_purification_transition(num_system=64, graph_family="cycle", rates="0.05, 0.10, 0.15, 0.20, 0.25, 0.30", trajectories=2000)`
  * `study_erasure_protection(num_system=32, measurement_rate=0.1, trajectories=500)`
* Resources: `docs://sparsegf2/quickstart` (a short Markdown guide for the agent:
  conventions for seeds, qubit layout, censoring) and `docs://sparsegf2/capabilities`
  (the `capabilities` overview as JSON).

### Outputs: summaries for the model, arrays on disk

Tool results are compact JSON summaries of at most 8000 characters, so that they fit
comfortably in the model's context: means and standard errors at no more than 50
checkpoints, P_C(m) curves for the erasure probes, purification statistics. The full
arrays are never sent to the model. Pass `output_dir` and they are written to disk
(`result.npz` plus `spec.json` and `provenance.json`), where you load them in Python for
the real analysis (section 5.6).

---

## 2. Install

Use the virtual environment from section 1 of [`GETTING_STARTED.md`](GETTING_STARTED.md)
(a `.venv` in the repository root) and add the `mcp` extra. From scratch:

```sh
git clone https://github.com/sjbevins/SparseGF2
cd SparseGF2
python -m venv .venv                            # Python 3.11 or newer
source .venv/bin/activate                       # Windows: .venv\Scripts\activate
pip install -e ".[mcp]"                         # the simulator plus the MCP server
```

The `mcp` extra adds the MCP Python SDK (`mcp>=2.2`) and `pydantic`; the simulator itself
never imports them. If you also want the test suite, install `".[test,mcp]"` (or
`".[dev]"` for everything).

Check that the command exists and starts:

```console
$ sparsegf2-mcp --help
usage: sparsegf2-mcp

Serve SparseGF2 over MCP (stdio). Register it with:
  claude mcp add sparsegf2 -- sparsegf2-mcp
$ which sparsegf2-mcp
/home/you/SparseGF2/.venv/bin/sparsegf2-mcp
```

(On Windows, `where sparsegf2-mcp` prints the path, ending in
`.venv\Scripts\sparsegf2-mcp.exe`.) Note the absolute path: clients that do not start
from your shell, such as Claude Desktop, need it.

Running `sparsegf2-mcp` with no arguments in a terminal starts the server and waits
silently for a client on stdin. That is normal; press Ctrl-C to stop it. You never need
to start it by hand: the client does.

Two other ways to start the same server, useful if the script is not on `PATH`:
`sparsegf2 mcp` and `python -m sparsegf2.mcp_server`.

---

## 3. Connect it to a client

### Claude Code

**In this repository, nothing to do.** The repository root holds a
[`.mcp.json`](../.mcp.json) that registers the server for Claude Code:

```json
{
  "mcpServers": {
    "sparsegf2": {
      "command": "sparsegf2-mcp",
      "args": []
    }
  }
}
```

Start Claude Code in the repository root with the virtual environment active (so that
`sparsegf2-mcp` is on `PATH`):

```sh
cd SparseGF2
source .venv/bin/activate
claude
```

Claude Code asks you once to approve the project's MCP server. Type `/mcp` inside the
session to see whether `sparsegf2` is connected and to list its tools. The repository
also ships an agent skill, [`.claude/skills/sparsegf2/SKILL.md`](../.claude/skills/sparsegf2/SKILL.md),
which tells Claude when and how to use the simulator.

**Anywhere else, or without activating the environment**, register the server yourself
with the absolute path from `which sparsegf2-mcp`:

```sh
claude mcp add sparsegf2 -- /home/you/SparseGF2/.venv/bin/sparsegf2-mcp
```

`--scope` chooses where the registration is stored: `local` (the default: only you, only
the current project), `project` (written to `.mcp.json` in the current project, shared
through version control) or `user` (you, in every project). To limit the server's CPU
threads (see section 6), pass an environment variable:

```sh
claude mcp add --scope user sparsegf2 --env NUMBA_NUM_THREADS=8 -- /home/you/SparseGF2/.venv/bin/sparsegf2-mcp
```

`claude mcp list` shows the registered servers. In a session, the tools appear as
`mcp__sparsegf2__run_ensemble` and so on. Claude Code asks for permission before it
calls a tool unless you have allowed it; adding `mcp__sparsegf2` to the permission allow
list allows all of this server's tools.

### Claude Desktop

Open the configuration file from the app (Settings, Developer, Edit Config). The file is
`claude_desktop_config.json`, in `~/Library/Application Support/Claude/` on macOS and in
`%APPDATA%\Claude\` on Windows. Add the server with the absolute command path, because
Desktop does not start from your shell and does not see your virtual environment:

```json
{
  "mcpServers": {
    "sparsegf2": {
      "command": "/Users/you/SparseGF2/.venv/bin/sparsegf2-mcp",
      "args": [],
      "env": {"NUMBA_NUM_THREADS": "8"}
    }
  }
}
```

The `env` block is optional. On Windows the command looks like
`"C:\\Users\\you\\SparseGF2\\.venv\\Scripts\\sparsegf2-mcp.exe"` (backslashes doubled in
JSON). Quit Claude Desktop completely and start it again; the server's tools then appear
in the chat's tools menu. Use absolute paths (or paths starting with `~`) for every
`output_dir` in Desktop; section 6 explains why.

### Any other MCP client

The server is a standard stdio server. Give your client the command
`/absolute/path/to/sparsegf2-mcp` with no arguments (or `sparsegf2 mcp`, or
`python -m sparsegf2.mcp_server` with that environment's `python`). It needs no
network access, no API keys and no configuration. See your client's documentation for
where stdio servers are registered.

---

## 4. Check that it works without an agent

This script plays the role of the client: it starts `sparsegf2-mcp` over stdio with the
MCP Python SDK (installed by the `mcp` extra), lists the tools and calls two of them. It
is the same pattern the server's own tests use
([`tests/test_mcp_server.py`](../tests/test_mcp_server.py)). Save it as `check_mcp.py`
and run it with the virtual environment active:

```python
"""Start sparsegf2-mcp over stdio, list its tools and call two of them."""

import asyncio
import json
import shutil

import mcp

SPEC = {
    "num_system": 16,
    "graph": {"family": "cycle"},
    "layers": 32,
    "measurement_rate": 0.2,
    "reference": "single",
    "checkpoints": 4,
    "probes": [{"kind": "code_dimension"}],
    "stop_when_purified": True,
}


async def main():
    command = shutil.which("sparsegf2-mcp")
    print("server command:", command)
    params = mcp.StdioServerParameters(command=command, args=[])
    async with mcp.Client(params) as client:
        print("server:", client.server_info.name, client.server_info.version)
        tools = (await client.list_tools()).tools
        print("tools:", ", ".join(t.name for t in tools))

        caps = await client.call_tool("capabilities", {"section": "probes"})
        print("probes:", ", ".join(caps.structured_content["probes"]))

        run = await client.call_tool("run_ensemble", {"spec": SPEC, "trajectories": 50, "seed": 1})
        if run.is_error:
            raise SystemExit(run.content[0].text)
        print("run_ensemble (text the agent reads, first 600 chars):")
        print(run.content[0].text[:600])


asyncio.run(main())
```

```console
$ python check_mcp.py
server command: /home/you/SparseGF2/.venv/bin/sparsegf2-mcp
server: sparsegf2 3.0.0.dev9
tools: capabilities, inspect_circuit, run_circuit, run_ensemble, estimate_ensemble, start_ensemble_job, job_status, cancel_job, list_jobs, load_results
probes: code_dimension, code_snapshot, contiguous_distance, entropy, entropy_profile, erasure_first_failure, erasure_lost_profile, half_cut_entropy, lost_information, mutual_information, tripartite_information, uncorrectable_region_counts
run_ensemble (text the agent reads, first 600 chars):
{"num_trajectories":50,"num_system":16,"num_qubits":17,"layers":32,"measurement_rate":0.2,"seed":1,"elapsed_seconds":0.44495,"trajectories_per_second":112.372,"checkpoints":{"layers":[0,4,8,12,16,20,24,28,32],"num_total":9,"subsampled":false},"probes":{"code_dimension":{"mean":[1.0,0.54,0.32,0.22,0.16,0.12,0.04,0.04,0.04],"stderr":[0.0,0.0711996,0.0666395,0.059178,0.0523723,0.0464231,0.0279942,0.0279942,0.0279942]}},"purification":{"trajectories":50,"purified":48,"censored":2,"censored_fraction":0.04,"horizon_layer":32,"tau_mean_purified":8.83333,"tau_median_purified":8.0,"tau_min":4,"tau_max"
```

The very first simulation after installing compiles the Numba kernels; on the machine
used here the first `run_ensemble` call took 33 s with an empty kernel cache and 0.4 s
once the cache existed. So if the script seems to hang the first time, wait. The
compiled kernels are cached on disk, and this run also warms the cache for your agent.

Each result has two forms: `content[0].text`, the compact JSON the model reads, and
`structured_content`, the same data as a Python dict. A failed call has
`is_error=True` and a message saying what to change.

---

## 5. A first session

Now open your agent and ask for things in plain language. Below, each request is
followed by the tool calls it should lead to, with their exact JSON arguments, and the
real result of those calls (from the same server, driven by a script like the one in
section 4). Your agent may choose slightly different arguments; it reads the tool
descriptions and the quickstart resource, and you can always tell it exactly what you
want.

A good first message is simply: *"Read the SparseGF2 quickstart and tell me what the
simulator can do."* The agent then reads `docs://sparsegf2/quickstart` and calls
`capabilities`.

### 5.1 Inspect and run a Stim circuit

> *"Check this circuit, then run it once with seed 1 and give me the entropy of qubit 2
> and the expectation of Z1 Z2 after the TICK and at the end:*
> `H 0` / `CX 0 1` / `CX 1 2` / `TICK` / `M 0 1 2`. *Then sample it 1000 times."*

```text
inspect_circuit {"circuit": "H 0\nCX 0 1\nCX 1 2\nTICK\nM 0 1 2"}
```
```text
{"valid":true,"num_qubits":3,"num_measurements":3,"num_detectors":0,"num_observables":0,"num_ticks":1,"num_operations":7,"repeat_blocks":0,"instruction_histogram":{"M":3,"CX":2,"H":1,"TICK":1},...,"stim_compatible":true,...}
```

```text
run_circuit {"circuit": "H 0\nCX 0 1\nCX 1 2\nTICK\nM 0 1 2", "shots": 1, "seed": 1,
             "observables": [{"kind": "entropy", "region": [2]},
                             {"kind": "expectation", "pauli": "Z1*Z2"}],
             "observe_ticks": true}
```
```text
{"num_qubits":3,...,"shots":1,"phases":true,"seed":1,"random_fraction":0.333333,"random":"100","measurements":"111","observables":{"entropy":0,"expectation":1},"tick_observables":{"ticks":[0],"values":{"entropy":[1],"expectation":[1]},...}}
```

The GHZ state has one bit of entanglement across qubit 2 after the TICK (`entropy` 1);
after the measurements the state is a product state (`entropy` 0), and Z1 Z2 is +1
throughout. Only the first measurement was random (`"random":"100"`).

```text
run_circuit {"circuit": "H 0\nCX 0 1\nCX 1 2\nTICK\nM 0 1 2", "shots": 1000, "seed": 1}
```
```text
{...,"shots":1000,"phases":true,"seed":1,"random_fraction":0.333333,"measurement_mean":0.495,"measurement_means":[0.495,0.495,0.495],"first_shots":["000","111","000","111"]}
```

### 5.2 A purification ensemble

> *"Run 2000 trajectories of a monitored circuit on a ring of 32 qubits at measurement
> rate 0.1 for 256 layers, with one reference qubit, stopping each trajectory when the
> reference purifies. Seed 7. Estimate the time first and save the results under
> ~/sparsegf2-runs/purification/n32_p0.10."*

```text
estimate_ensemble {"spec": {"num_system": 32, "graph": {"family": "cycle"}, "layers": 256,
                            "measurement_rate": 0.1, "reference": "single",
                            "checkpoints": "every_layer",
                            "probes": [{"kind": "code_dimension"}],
                            "stop_when_purified": true},
                   "trajectories": 2000}
```
```json
{"estimated_seconds":1.48454,"trajectories":2000,"result_megabytes":4.176,"threads":4,"fits_sync_limit":true,"sync_limit_seconds":60.0}
```

```text
run_ensemble {"spec": {...same spec...}, "trajectories": 2000, "seed": 7,
              "output_dir": "~/sparsegf2-runs/purification/n32_p0.10"}
```
```text
{"num_trajectories":2000,"num_system":32,"num_qubits":33,"layers":256,"measurement_rate":0.1,"seed":7,
 "elapsed_seconds":0.501117,"trajectories_per_second":3991.08,
 "checkpoints":{"layers":[0,5,10,16,21,...,246,251,256],"num_total":257,"subsampled":true},
 "probes":{"code_dimension":{"mean":[1.0,0.7705,0.7025,0.685,...,0.603,0.602],"stderr":[0.0,0.00940527,...,0.010948]}},
 "purification":{"trajectories":2000,"purified":796,"censored":1204,"censored_fraction":0.602,"horizon_layer":256,
   "tau_mean_purified":26.2651,"tau_median_purified":4.0,"tau_min":1,"tau_max":256,
   "tau_quantiles_purified":{"p10":1.0,"p25":2.0,"p75":11.0,"p90":103.0},"tau_median_all":null,
   "survival":{"layers":[0,5,10,...,256],"fraction_unpurified":[1.0,0.7705,0.7025,...,0.602]}},
 "final_layer_mean":164.565,"measurements_per_trajectory":526.674,
 "notes":["stop_when_purified: code probes take their exact values for a purified trajectory (k = 0, nothing lost); entanglement probes average only trajectories still running (see 'count')"],
 "files":{"result":"/home/you/sparsegf2-runs/purification/n32_p0.10/result.npz","spec":".../spec.json","provenance":".../provenance.json"}}
```

How to read it: 257 checkpoints were recorded and 50 are shown (`subsampled`); the full
set is in `result.npz`. 60% of the trajectories never purified within 256 layers
(`censored_fraction` 0.602), so the median purification time over all trajectories is
undefined (`tau_median_all` is `null`) and the agent must not average `tau` with the
censored ones. The survival curve is the honest summary.

### 5.3 An erasure study with graph collections

This is a scaled-down version of an erasure-protection study: 20 Watts-Strogatz graphs
(beta = 1) with 2 circuits each, the code dimension at 27 geometrically spaced depths,
and the expensive erasure probes on their own sparser schedule.

> *"Set up an erasure-protection run: 32 qubits, 20 Watts-Strogatz graphs with k = 4,
> beta = 1, graph seed 11, 2 circuits per graph, random-edge gates, measurement rate
> 0.05, purification reference, 1000 layers. Record the code dimension at these depths:
> 1, 2, 3, 4, 5, 7, 9, 11, 14, 17, 22, 28, 36, 45, 57, 73, 92, 117, 149, 189, 240, 304,
> 386, 489, 621, 788, 1000. At depths 10, 100 and 1000 only, record 100 first-failure
> samples and the exact counts of uncorrectable 1-, 2- and 3-qubit erasures, and save the
> codes at the final layer. Circuit seed 1. Estimate first, then save to
> ~/sparsegf2-runs/erasure/n32_beta1_p0.05."*

The spec the agent builds:

```json
{
  "num_system": 32,
  "graph": {"family": "watts_strogatz", "k": 4, "beta": 1.0, "seed": 11, "realizations": 20},
  "circuits_per_graph": 2,
  "layers": 1000,
  "measurement_rate": 0.05,
  "schedule": "random_edges",
  "reference": "purification",
  "checkpoints": [1, 2, 3, 4, 5, 7, 9, 11, 14, 17, 22, 28, 36, 45, 57, 73, 92, 117, 149,
                  189, 240, 304, 386, 489, 621, 788, 1000],
  "probes": [
    {"kind": "code_dimension"},
    {"kind": "erasure_first_failure", "samples": 100, "checkpoints": [10, 100, 1000]},
    {"kind": "uncorrectable_region_counts", "max_size": 3, "checkpoints": [10, 100, 1000]},
    {"kind": "code_snapshot", "checkpoints": "final"}
  ]
}
```

What each piece means:

* `"realizations": 20` inside `graph` builds a collection of 20 graphs (graph `i` from
  seed `[11, i]`); `"circuits_per_graph": 2` runs 2 random circuits on each, so trajectory
  `t` runs on graph `t // 2` and there are 40 trajectories. `trajectories` is therefore
  left out of the calls below. Reuse the same graph seed at every measurement rate to
  run the same graphs everywhere.
* The spec's `checkpoints` are the default schedule for every probe; a probe's own
  `"checkpoints"` overrides it. Give the expensive probes (many samples, exhaustive
  counts, stored codes) a sparse schedule of their own.
* `erasure_first_failure` stores, for each of 100 random orderings of the qubits, the
  smallest number of erased qubits that leaks logical information (n + 1 if the code has
  no logical qubits).
* `uncorrectable_region_counts` with `max_size` 3 counts, by exhaustion, how many m-qubit
  erasures are uncorrectable for m = 1, 2, 3.
* `code_snapshot` saves the stabilizer generators of each code, so that any other
  erasure quantity can be computed later. The codes are only kept with `output_dir`.

```text
estimate_ensemble {"spec": {...the spec above...}}
```
```json
{"estimated_seconds":0.167286,"trajectories":40,"result_megabytes":0.1296,"threads":4,"fits_sync_limit":true,"sync_limit_seconds":60.0}
```

```text
run_ensemble {"spec": {...the spec above...}, "seed": 1,
              "output_dir": "~/sparsegf2-runs/erasure/n32_beta1_p0.05"}
```
```text
{"num_trajectories":40,"num_system":32,"num_qubits":64,"layers":1000,"measurement_rate":0.05,"seed":1,"randomness_contract":2,...,
 "checkpoints":{"layers":[1,2,3,4,5,7,9,...,621,788,1000],"num_total":27,"subsampled":false},
 "probes":{
  "code_dimension":{"mean":[30.35,29.025,27.35,...,10.65,10.375],"stderr":[0.197906,...,0.141817]},
  "erasure_first_failure":{"what":"P_C(m): fraction of random m-qubit erasures that are uncorrectable vs m (index), estimated from the first-failure samples, among trajectories that hold a code (k > 0); count_with_code says how many; at_layer is the last evaluation with count_with_code > 0, if there is one",
    "vector":true,"at_layer":1000,"count":40,"count_with_code":40,
    "mean":[0.0,0.0085,0.0205,0.03525,0.05875,0.1025,0.16325,0.265,0.41975,0.6145,0.809,0.93225,0.98575,0.9975,1.0,...,1.0],
    "stderr":[...],
    "vs_depth":{"layers":[10,100,1000],"count":[40,40,40],"count_with_code":[40,40,40],"index":[0,2,4,...,32],"mean":[[0.0,0.86125,0.9835,...],[0.0,0.09375,0.2575,...],[0.0,0.0205,0.05875,...]]}},
  "uncorrectable_region_counts[3]":{"what":"exact P_C(m): fraction of all m-qubit erasures that are uncorrectable vs m (index), among trajectories that hold a code (k > 0); count_with_code says how many; at_layer is the last evaluation with count_with_code > 0, if there is one",
    "vector":true,"at_layer":1000,"count":40,"count_with_code":40,"mean":[0.0,0.00859375,0.0199093,0.0362752],"stderr":[...],
    "vs_depth":{"layers":[10,100,1000],...,"mean":[[0.0,0.60625,0.858115,0.951482],[0.0,0.0375,0.0859375,0.154703],[0.0,0.00859375,0.0199093,0.0362752]]}},
  "code_snapshot":{"mean":[21.625],"stderr":[0.141817],"layers":[1000]}},
 "purification":{"trajectories":40,"purified":0,"censored":40,"censored_fraction":1.0,...},
 "final_layer_mean":1000.0,"measurements_per_trajectory":1602.9,
 "truncated":true,
 "notes":["code_snapshot: the mean number of stabilizer generators (n - k); the codes themselves are saved with output_dir (EnsembleResult.code(t, c) in Python)",
          "vector probes are subsampled: curves to at most 65 sizes m (listed in 'index'), vs_depth grids to at most 8 evaluations by 17 sizes; pass output_dir or read the saved files for every value"],
 "files":{"result":"/home/you/sparsegf2-runs/erasure/n32_beta1_p0.05/result.npz",...}}
```

The two erasure probes are summarized as P_C(m) curves at the last of their
checkpoints (index = m), with a coarse depth-by-size grid in `vs_depth`; `layers`
appears in a probe's summary when its schedule differs from the spec's. Both curves
average only the trajectories that hold a code (`count_with_code`: all 40 here, since
none purified). The first-failure probe tells these from its own samples (a code with
k = 0 gives D = n + 1). The exact counts cannot tell k = 0 from a large distance, so they
take the code holders from a `code_dimension` probe evaluated at each of their layers or,
failing that, from such an `erasure_first_failure` probe. Here `code_dimension` misses
depths 10 and 100 (the spec's checkpoints skip them), so the first-failure probe on the
same schedule is used. Without either, the counts curve would average every trajectory
and a note would say so.
`truncated` and the "vector probes are subsampled" note are routine here: the grid
shows 17 of the 33 sizes. The code dimension falls from 32 to about 10 over 1000
layers, and the code at depth 1000 protects against small erasures far better than at
depth 10: about 0.86% of 1-qubit erasures are uncorrectable at depth 1000 (exact
count), against 61% at depth 10.

### 5.4 Long runs: background jobs

`run_ensemble` refuses any run estimated above 60 s (its `max_seconds`) and says to use a
job instead. A job runs in a background thread of the server, saves every chunk as soon
as it finishes, and gives results identical to one `run_ensemble` call with the same spec
and seed.

> *"Run the same erasure spec with 200 graphs instead of 20 as a background job in
> ~/sparsegf2-runs/erasure/n32_beta1_p0.05_G200, in chunks of 100 trajectories. Tell me
> when it is done and summarize it."*

```text
start_ensemble_job {"spec": {...same, with "realizations": 200...}, "seed": 1, "chunk_size": 100,
                    "output_dir": "~/sparsegf2-runs/erasure/n32_beta1_p0.05_G200"}
```
```text
{"job_id":"b967c4d83b75","status":"running","trajectories_done":0,"trajectories_total":400,"progress":0.0,"resumed_from":0,...,"chunk_size":100,"seed":1,"output_dir":"/home/you/sparsegf2-runs/erasure/n32_beta1_p0.05_G200"}
```

```text
job_status {"job_id": "b967c4d83b75", "wait_seconds": 60}
```
```text
{"job_id":"b967c4d83b75","status":"completed","trajectories_done":400,"trajectories_total":400,"progress":1.0,...,"trajectories_per_second":1222.19,...,"next_step":"load_results('/home/you/sparsegf2-runs/erasure/n32_beta1_p0.05_G200') for the summary"}
```

```text
load_results {"path": "~/sparsegf2-runs/erasure/n32_beta1_p0.05_G200"}
```
```text
{"num_trajectories":400,"num_system":32,...,"probes":{"code_dimension":{"mean":[30.3475,28.8825,...,10.255],...},"erasure_first_failure":{...,"count":400,"mean":[0.0,0.002925,0.00735,...]},...},...,"path":"...","spec":{...},"trajectories_requested":400}
```

`job_status` with `wait_seconds` blocks for up to 60 s per call, so the agent polls
cheaply; `list_jobs` shows every job of this server process, and `cancel_job` stops one
after its current chunk. `load_results` also works while a job is running or after a
cancel: it reads the chunks saved so far.

### 5.5 The built-in prompts

The two prompts are complete study plans the agent follows step by step. In Claude Code,
MCP prompts appear as slash commands (type `/` and look for the `sparsegf2` entries); in
other clients look for a prompts or attachments menu. You can also just ask: *"Use the
SparseGF2 prompt study_erasure_protection with num_system 64."* Its text, as the server
returns it:

```text
Study the erasure protection of the code generated by monitored random Clifford dynamics.

Setup: 64 system qubits on a cycle, reference="purification" (every system
qubit entangled with its own reference, so the code starts with 64 logical
qubits), measurement_rate=0.1, layers = 4 * num_system, checkpoints=16,
probes: code_dimension, erasure_first_failure (samples 100) and
uncorrectable_region_counts (max_size 2). Trajectories: 500.

Steps:
1. Call estimate_ensemble, then run_ensemble (or start_ensemble_job if too slow) with an
   output_dir.
2. Report how the code dimension k(t) decays with depth and, at the final checkpoint,
   P_C(m), ...
```

`study_purification_transition` sweeps the measurement rate, runs one job per rate and
reports censored fractions, purification-time quantiles and survival curves.

### 5.6 The full analysis in Python

The summaries are for steering the work. For figures, load the saved `result.npz` in
Python (the same virtual environment) and use `sparsegf2.analysis`. This computes the
code rate, the probability that a code has logical qubits, Phi(e) (the probability that
erasing each qubit independently with probability e is uncorrectable) and the erasure
tolerance e_0.10, at each depth of the erasure probes, and decodes one saved code:

```python
from pathlib import Path

import numpy as np

from sparsegf2 import analysis, mipt

res = mipt.EnsembleResult.load(Path("~/sparsegf2-runs/erasure/n32_beta1_p0.05_G200/result.npz").expanduser())
n = res.spec.num_system
k = res.values["code_dimension"]                       # (trajectories, 27)
first = res.values["erasure_first_failure"]            # (trajectories, 3, 100)
counts = res.values["uncorrectable_region_counts[3]"]  # (trajectories, 3, 3)
print("trajectories:", res.num_trajectories, "graphs:", np.unique(res.graph_index).size)
print("code rate r(T) at", res.layers("code_dimension")[-3:], ":", (k.mean(axis=0) / n)[-3:])

kk = k[:, np.searchsorted(res.layers("code_dimension"), res.layers("erasure_first_failure"))]
for c, T in enumerate(res.layers("erasure_first_failure")):
    phi = analysis.erasure_failure_probability(
        first[:, c], n, [0.05, 0.10, 0.20], exact_counts=counts[:, c]).mean(axis=0)
    has_code = kk[:, c] > 0
    e10 = analysis.erasure_tolerance(first[has_code, c], n, 0.10, exact_counts=counts[has_code, c])
    print(f"T={T:5d}  q+={has_code.mean():.3f}  Phi(0.05,0.10,0.20)={np.round(phi, 4)}  e_0.10={e10:.4f}")

code = res.code(0, 0)          # trajectory row 0, first (only) snapshot evaluation
print("code of trajectory 0 at T =", res.layers("code_snapshot")[0], ":",
      code.num_qubits, "qubits,", code.num_logical, "logical; uncorrectable counts m=1..3:",
      code.uncorrectable_counts(3), "(probe recorded:", counts[0, -1], ")")
```

```text
trajectories: 400 graphs: 200
code rate r(T) at [ 621  788 1000] : [0.345625   0.33398438 0.32046875]
T=   10  q+=1.000  Phi(0.05,0.10,0.20)=[0.6373 0.8772 0.9886]  e_0.10=0.0054
T=  100  q+=1.000  Phi(0.05,0.10,0.20)=[0.0845 0.2262 0.6141]  e_0.10=0.0567
T= 1000  q+=1.000  Phi(0.05,0.10,0.20)=[0.0086 0.0371 0.2495]  e_0.10=0.1444
code of trajectory 0 at T = 1000 : 32 qubits, 9 logical; uncorrectable counts m=1..3: [0 0 0] (probe recorded: [0. 0. 0.] )
```

`exact_counts` switches on the hybrid estimator: exact P_C(m) for m <= 3 from the
exhaustive counts and the first-failure samples above that. `erasure_tolerance` is
computed over the codes that still have logical qubits (`has_code`), as the definition
of e_delta asks. `res.code(t, c)` returns a `StabilizerCode`, which can recompute any
erasure quantity offline (`first_failure_sizes`, `uncorrectable_counts`,
`lost_information`, ...). [`GETTING_STARTED.md`](GETTING_STARTED.md) shows the complete
analysis workflow, including heatmaps over measurement rate and depth.

### 5.7 From a pilot to a full campaign

The MCP tools take the spec inline, as JSON written by the model. That is ideal for
exploring, pilots and runs of moderate size. Two things to know before asking an agent
for a full-scale run:

* **Ask for the estimate first.** For example, one point of a 256-qubit study (1000
  graphs, one circuit each, 10^6 layers, the code dimension at the 5957 distinct
  depths of `geomspace(1, 1e6, 10000)`, 100 first-failure samples and exhaustive
  counts up to 3 qubits), `estimate_ensemble` returned, on the 4-core machine used here:
  about 251,000 s and 4973 MB with `code_dimension` and the erasure probes at all 5957
  depths (codes saved at the final layer only), and about 27,600 s and 208 MB with the
  erasure probes on a 175-depth schedule of their own (saving codes at all 175 depths
  as well would raise this to 3060 MB). The estimate is deliberately pessimistic and scales with the number of
  cores. Results above 8 GB are refused outright, with a message suggesting fewer
  checkpoints for the large probes or several output directories.
* **Long checkpoint lists are long JSON.** 5957 layers are about 40 kB of text for the
  model to write, and a run of many hours should not depend on a chat client staying
  open. For the full campaign, run from Python or the command line on your workstation
  or cluster: section 5 of [`GETTING_STARTED.md`](GETTING_STARTED.md) has a complete
  script for one (beta, p) point, chunking and a Slurm array, and section 6.4 the
  command line (`sparsegf2 ensemble spec.json --output-dir DIR --chunk-size N`,
  resumable by running the same command again). The command line's output directory has
  the same layout as a job's, so the agent can still summarize it with `load_results`,
  and you analyse it in Python as in section 5.6. An agent with shell access can also
  write such a script for you.

---

## 6. Files, paths and long runs

**Where `output_dir` goes.** The server expands `~` and resolves the path with
`Path(output_dir).expanduser().resolve()`, so a relative path is relative to the
**server process's working directory**, not yours and not the model's idea of it. Claude
Code normally starts the server in the directory where you started `claude`; Claude
Desktop starts it in a directory you did not choose. Use absolute paths or `~/...` for
every `output_dir` (and every `load_results` path); the `files` entry of each result
reports the absolute paths actually written. Parent directories are created as needed.

**What is in an output directory.**

```text
~/sparsegf2-runs/erasure/n32_beta1_p0.05_G200/
  result.npz        the full EnsembleResult (all probe arrays, layers, seeds, codes)
  spec.json         {"spec": ..., "trajectories": ..., "seed": ...}: replaying it repeats the
                    run, if this version has the randomness contract in provenance.json
  provenance.json   package version, randomness contract, Python, numpy and numba
                    versions, git commit, platform, seed, UTC timestamp, command; one
                    entry per resume under "resumes"
  chunks/           only while a job runs or after a cancel: traj_000000000000.npz, ...
                    (merged into result.npz and deleted when the job completes)
```

`run_ensemble` refuses an `output_dir` that already holds a run, so a finished run is
never overwritten; choose a new directory per run (for example one per measurement rate).
Load results in Python with `mipt.EnsembleResult.load(path)`; several result files of the
same spec and seed merge with `mipt.EnsembleResult.concatenate`.

**Resuming and extending jobs.** Call `start_ensemble_job` again with the same spec and
the same `output_dir`; `seed` may be omitted (the stored one is used). Saved trajectories
are kept and the job continues from there (`resumed_from` in the status). A larger
`trajectories` extends a finished run; for a graph collection the maximum is
`realizations x circuits_per_graph`, so start with fewer trajectories and leave
`trajectories` out later to run the rest. A directory holding a different spec, a
different seed, or a run started with a different SparseGF2 version or randomness
contract (`mipt.RANDOMNESS_CONTRACT`) is refused with a message.

**Jobs live inside the server process.** The client starts the server, and when the
client exits (you close Claude Code or Claude Desktop, or the client restarts the server)
the server and its running jobs stop. Saved chunks stay on disk. In the next session the
old `job_id` is unknown (the error message says so), and the remedy is to resume: ask
the agent to call `start_ensemble_job` with the same spec and `output_dir`. If the agent
no longer has the spec, `load_results` on the directory returns it under `"spec"` (the
file `spec.json` holds it too). Verified while writing this guide: a job of 100 out of
400 trajectories in one server process, then, in a new process, `job_status` of the old
id failed with `unknown job_id ... (jobs do not survive a restart; resume by calling
start_ensemble_job with the same output_dir)`, and `start_ensemble_job` with the same
spec and directory resumed from 100 and completed 400. For runs of many hours, keep the
client open, or use the command line (section 5.7), which does not depend on a client.

**The first call compiles.** Each Numba kernel compiles on its first use and is then
cached on disk, so the first call of a new kind of run is slow (33 s for the first
`run_ensemble` here with an empty cache) and later ones are fast. Erasure probes and
`run_circuit` use further kernels, which compile on their first use. To avoid a slow
first call in an important session, warm up with a tiny call of the same kind, or run
the script of section 4.

**Limits.**

| Limit | Value | What happens |
|---|---|---|
| tool output | 8000 characters | long arrays are subsampled (at most 50 checkpoints shown, P_C(m) at most 65 sizes) and marked; full data via `output_dir` |
| synchronous ensemble | estimated above `max_seconds` (default 60) | `run_ensemble` refuses before running and points to `start_ensemble_job`; the agent may pass a larger `max_seconds` if you are willing to wait |
| result size | 8 GB | refused by `estimate_ensemble`, `run_ensemble` and `start_ensemble_job`, with advice |
| `job_status` wait | 0 to 60 s per call | poll again |
| graph collection | `realizations` at most 100,000 | error |

**Threads.** The engine runs trajectories in parallel on all cores. On a shared machine
set `NUMBA_NUM_THREADS` in the server's environment (the `env` block or `--env` of
section 3); `estimate_ensemble` reports the thread count the server uses as `threads`.
Only one ensemble computes at a time inside a server process; a second call waits for
the first.

---

## 7. Troubleshooting

**The client says the server failed to start, or shows no tools.** Run the exact command
from your client configuration in a terminal:

```sh
/home/you/SparseGF2/.venv/bin/sparsegf2-mcp --help
```

If that prints the usage text, the installation is fine and the problem is the client
configuration (usually a relative command or a typo in the path). If it prints an error,
fix that first. In Claude Code, `/mcp` shows the server's status; restart Claude
Desktop completely after every configuration change.

**`command not found: sparsegf2-mcp`, or the wrong Python.** The virtual environment is
not active, or the package was installed into another Python. Activate the environment
and check `which sparsegf2-mcp` and `python -c "import sparsegf2, sys; print(sparsegf2.__file__, sys.executable)"`.
Use the absolute path in client configurations. If you work from a different checkout,
reinstall with `pip install -e ".[mcp]"` from that checkout's root.

**`error: The SparseGF2 MCP server needs the optional 'mcp' extra`.** The MCP SDK is
missing: `pip install -e ".[mcp]"`. The server needs SDK version 2.2 or newer (it uses
the v2 `MCPServer` API); an older 1.x `mcp` package fails the same way, and the same
command upgrades it. Check with
`python -c "import importlib.metadata as m; print(m.version('mcp'))"`.

**The first call takes half a minute.** That is Numba compiling the kernels (section 6).
It happens once per installation or update; later calls load the cache. If your client
gives up on slow tool calls, warm up first with the script of section 4 or a tiny call.

**The machine becomes unresponsive during a run.** The engine uses every core; set
`NUMBA_NUM_THREADS` in the server's environment (section 3) and restart the server.

**`output_dir ... already holds a run`.** `run_ensemble` never overwrites. Use a new
directory, or `start_ensemble_job` to resume or extend the run that is there.

**The results are not where you expected.** A relative `output_dir` was resolved against
the server's working directory. Every result's `files` entry (and every job status's
`output_dir`) shows the absolute path; use absolute paths from now on.

**A tool call fails with a message.** Tool errors are written for the model to act on
(for example `unknown key 'layrs' ... layers`, or `this run is estimated at about N s,
above max_seconds=60. Use start_ensemble_job ...`). The agent normally fixes the
argument and retries; if it does not, the message tells you what to change.

---

## See also

* [`GETTING_STARTED.md`](GETTING_STARTED.md): installing SparseGF2 and running simulations
  from Python and the command line.
* [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md): how the simulator works.
* [`LLM_INTERFACE.md`](LLM_INTERFACE.md): the complete reference for the JSON interface
  (every tool argument, the summary format, the CLI).
* [`../AGENTS.md`](../AGENTS.md): the guide written for the agents themselves.
