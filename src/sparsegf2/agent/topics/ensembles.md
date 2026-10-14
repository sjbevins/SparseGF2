# Ensembles of random monitored circuits

`ensemble_run` simulates many trajectories of one protocol inside a single compiled kernel. The parameters are the `MonitoredCircuit` fields under their own names; `help_list(what="protocol_parameters")` lists them. `num_system`, `layers` and `measurement_rate` are required.

- **Setup.** `reference` picks the picture (`none`, `purification`, `single`), with `reference_partner` for `single`.
- **Scramble (optional).** `scramble_gates` random all-to-all two-qubit Cliffords, then `scramble_matchings` layers of gates on random perfect matchings, then `scramble_layers` gate-only layers on the graph.
- **Layer.** Gates on graph edges, placed by `schedule`: `matching` (edge colouring; brickwork on an even ring), `random_edges` (`gates_per_layer` edges drawn with replacement, weighted by `edge_weights` if given), or `all_edges`. Then each system qubit is measured in Z with probability `measurement_rate`.
- **Graph collections.** `graph_count` G graphs (graph i from seed `(graph_seed, i)`) times `circuits_per_graph` c circuits each. Trajectory t runs on graph `t // c`. Leave `trajectories` out to run all G x c; asking for more is an error.
- **Checkpoints.** `final` (default), `every_layer`, an integer stride, or a list of layers. Probes are recorded there, or at their own `@checkpoints`.
- **Estimates.** `estimate_only=true` returns a pessimistic runtime and the memory of the result without running.
- **Background.** Long runs become jobs; see `jobs`. Every chunk is saved as it finishes.

`ensemble_sweep` takes everything `ensemble_run` takes plus `grid`, for example `measurement_rate=[0.05,0.1,0.15]; num_system=[64,128]`: every combination runs as its own chunked ensemble in `output_dir/<point>/`, always as a background job. Points share graph seeds and circuit seeds. Compare the points with `analysis_compare`.

Example: `ensemble_run(num_system=64, layers=512, measurement_rate=0.1, reference="single", probes=["code_dimension"], trajectories=1000, seed=7)`.
