# Recipes

## 1. A purification transition

1. `ensemble_sweep(num_system=32, layers=256, reference="single", reference_partner=-1, schedule="matching", stop_when_purified=true, checkpoints="every_layer", probes=["code_dimension"], trajectories=500, seed=1, grid="measurement_rate=[0.05,0.1,0.15,0.2]; num_system=[32,64]", output_dir="runs/pt")`. It returns a job id.
2. `job_status(job_id, wait_seconds=30)` until it reports completed.
3. `analysis_compare(source="runs/pt", metrics=["censored_fraction", "survival(at=128)", "tau_median"])`. The rate where the curves for different sizes cross locates the transition.

## 2. An erasure study on a Watts-Strogatz collection

1. `ensemble_run(num_system=64, graph="watts_strogatz", graph_k=4, graph_beta=0.1, graph_count=50, graph_seed=0, circuits_per_graph=4, layers=256, measurement_rate=0.05, schedule="random_edges", reference="purification", checkpoints=32, probes=["code_dimension", "erasure_first_failure(samples=512)", "uncorrectable_region_counts(2)", "code_snapshot@final"], seed=1, output_dir="runs/erasure")`.
2. `analysis_erasure(source="runs/erasure", rates=[0.05,0.1,0.2], delta=0.01)`.
3. Pick an interesting trajectory (say the smallest first-failure size) and `traj_run` it with the same protocol and seed, `trajectory=<t>`, to store its code.
4. `code_analyze(source="runs/erasure/<trajectory>", properties=["distance_bounds(samples=10000, max_size=3)", "failure_probability(rates=[0.05,0.1], delta=0.01)"])` for sharper numbers with more orderings.

## 3. Checking a circuit

1. `circuit_inspect(circuit="H 0\nCX 0 1\nTICK\nM 0 1")` for counts, or the line number of any error.
2. `circuit_run(circuit=..., shots=1000, seed=1, observables=["entropy(0:1)"])` for means and flip rates.
3. `circuit_crosscheck(circuit=..., compare="both", seed=1)` to compare against Stim. It needs a noise-free circuit and Stim installed.
