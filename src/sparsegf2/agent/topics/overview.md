# SparseGF2 overview

SparseGF2 is a fast stabilizer (Clifford) circuit simulator built for monitored random Clifford circuits: measurement-induced and purification transitions, the codes these circuits generate dynamically, and how well those codes survive erasures. It reads Stim's circuit language, and its results are checked against Stim in the test suite.

The tools come in stacks, one per kind of work. A server serves one stack, chosen at launch.

- `help`: this guidance (`help_topic`) and exact names (`help_list`).
- `state`: stabilizer states built gate by gate and queried.
- `circuit`: Stim-language circuits: inspect, run, convert, cross-check against Stim.
- `ensemble`: ensembles of random monitored circuits and sweeps over their parameters.
- `traj`: single trajectories, with the code stored, and replay with new probes.
- `code`: analysis of any stabilizer code, such as erasure properties.
- `graph`: interaction graphs: build, save and measure them.
- `analysis`: summaries, probe statistics, erasure and purification analysis, comparison of runs.
- `result`: list, read and merge saved files.
- `job`: status, cancel and list of background work.
- `plot`: figures (needs matplotlib).

## How to start

1. Read `help_topic(topic="conventions")` once: qubit numbering, layers, seeds and units.
2. Call `help_list(what="tools")` to see the tools of this server, and `help_list(what="probes")` or `help_list(what="observables")` for exact names.
3. Structured arguments such as probes are call strings, for example `entropy(0:8)`. See the `grammar` topic.

Work that takes long (large ensembles, sweeps) returns a job id instead of a result. Poll it with `job_status`; see the `jobs` topic. The `recipes` topic has three worked sequences of calls.
