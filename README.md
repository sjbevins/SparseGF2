# SparseGF2

**Fast stabilizer simulation for monitored random Clifford circuits, checked against Stim.**

SparseGF2 simulates Clifford circuits with mid-circuit measurements, built for
the physics of monitored random circuits: measurement-induced and purification
transitions, the error-correcting codes these circuits generate as they run,
and their protection against erasures. It also runs general circuits written
in Stim's circuit language, surface codes included, and every operation is
tested against Stim.

The current version is **SparseGF2 v3**, in [`v3/`](v3/README.md) (import name
`sparsegf2_v3`; it will take over the `sparsegf2` name as 3.0). Its
[README](v3/README.md) has the quickstart, the physics examples, the
performance comparison and how it works.

> The old package, SparseGF2 v2 (import name `sparsegf2`), still lives at the
> repository root; its documentation is [`README_v2.md`](README_v2.md).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="v3/docs/figures/circuit-dark.png">
  <img alt="A monitored random Clifford circuit: random two-qubit Cliffords in a brickwork, Z measurements at rate p, and a reference qubit Bell-paired with the system" src="v3/docs/figures/circuit-light.png">
</picture>

## Physics in a few lines

```python
from sparsegf2_v3 import graphs, mipt

n = 128
spec = mipt.MonitoredCircuit(
    num_system=n,
    edges=[graphs.watts_strogatz(n, k=4, beta=0.1, seed=(1, g)) for g in range(2500)],
    circuits_per_graph=4,             # 2500 graphs x 4 random circuits each
    layers=8 * n,
    measurement_rate=0.25,
    schedule="random_edges",          # n/2 random graph edges per layer
    reference="single",               # one reference qubit, Bell-paired ...
    reference_partner=n - 1,          # ... with system qubit n-1
    scramble_matchings=32,            # 32 layers of random all-to-all matchings first
    checkpoints="every_layer",
    stop_when_purified=True,
)
result = spec.run(seed=7)                         # all 10,000 trajectories, parallel over all cores
result.purification_layer                         # tau per trajectory (-1 = censored)
layers, surviving = result.survival()             # P(tau > t)
spec.circuit(trajectory=3, seed=7)                # the exact circuit trajectory 3 ran
```

The same engine records entanglement (`entropy`, `half_cut_entropy`,
`mutual_information`, `tripartite_information`, `entropy_profile`) and the
generated code (`code_dimension`, `contiguous_distance`, first-failure sizes
and exhaustive counts of uncorrectable erasures, the code itself), each probe
at the layers you choose. `sparsegf2_v3.analysis` turns the erasure probes
into the failure probability under independent erasure of each qubit and the
tolerable erasure rate; see
[Erasure protection of the generated code](v3/README.md#4-erasure-protection-of-the-generated-code).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="v3/docs/figures/transition-dark.png">
  <img alt="Tripartite information -I3 versus measurement rate for n = 32 to 256; the curves cross near p = 0.16" src="v3/docs/figures/transition-light.png">
</picture>

## Speed

On monitored random Clifford circuits (n qubits, depth 2n, one core each,
identical circuits, final states cross-checked), SparseGF2 v3 in phase-free
mode is **3–15× faster than Stim** for n ≥ 64, and about 3× faster with signs
tracked. Its ensemble engine generates and simulates fresh random circuits in
parallel: about **128,000 circuits per second at n = 16** and 1,700 at
n = 128 on four cores. Method, caveats and every number:
[v3 README, Performance](v3/README.md#performance).

## Install

Python 3.11 or newer.

```sh
git clone https://github.com/sjbevins/SparseGF2.git
cd SparseGF2
pip install -e "./v3[test]"              # add ,mcp for the MCP server: "./v3[test,mcp]"
cd v3 && python -m pytest                # the full suite; Stim is required for the tests
```

## For AI agents

SparseGF2 v3 ships an MCP server (`sparsegf2-mcp`, registered for Claude Code
by this repository's [`.mcp.json`](.mcp.json)), a JSON CLI (`sparsegf2 ... --json`)
and a JSON service layer (`sparsegf2_v3.api`). Agents should start with
[`v3/AGENTS.md`](v3/AGENTS.md).

## Where to go next

| | |
|---|---|
| [`v3/README.md`](v3/README.md) | quickstart, erasure workflow, figures, performance, how it works |
| [`v3/examples/`](v3/examples/) | complete study scripts: purification times, transition scans, erasure protection |
| [`v3/docs/DESIGN.md`](v3/docs/DESIGN.md) | architecture, decisions, roadmap |
| [`v3/docs/LITERATURE.md`](v3/docs/LITERATURE.md) | the literature behind the feature set |
| [`v3/benchmarks/`](v3/benchmarks/) | reproducible speed comparisons |
| [`README_v2.md`](README_v2.md) | the old v2 package |

## License

MIT; see [`LICENSE`](LICENSE).
