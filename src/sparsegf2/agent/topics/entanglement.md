# Entanglement

Entropies are in bits, as exact integers per trajectory. Probes (see `probes`):

- `entropy(region)`: the entropy of a qubit region such as `0:8`.
- `half_cut_entropy`: the entropy of half of the system, a single cheap number to watch versus depth.
- `mutual_information(a, b)`: I(A:B) between two regions, for example `mutual_information(a=0:4, b=8:12)`.
- `tripartite_information`: I3 of three regions. For brickwork on a ring, -I3 crosses near the critical measurement rate p_c ~ 0.16.
- `entropy_profile`: a vector, the entropy S of the first m qubits of `system_order`, m = 0..n.

Entanglement probes are identical with and without phases. They are cheap inside the engine: measuring the half-cut entropy after every layer adds only a fraction of a millisecond per layer at n = 1024.

With `stop_when_purified`, an entanglement probe is NaN after a trajectory has stopped; averages use only the trajectories still running, and the per-checkpoint `count` says how many there were.

## Typical use

1. Run `ensemble_sweep` over `measurement_rate` with probes `half_cut_entropy@every_layer` and `entropy_profile@final`.
2. Read the area-law and volume-law behaviour from `analysis_probe(probe="entropy_profile", statistic="mean")`, and compare sizes with `analysis_compare(metrics=["mean(half_cut_entropy)"])`.
3. For a single state, `state_query(state_id, ["entropy(0:8)"])` gives the same quantity gate by gate, and `circuit_run` with `observe_ticks=true` gives it after every TICK.
