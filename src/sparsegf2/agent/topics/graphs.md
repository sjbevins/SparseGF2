# Graphs

The interaction graph sets where gates may act. Every graph parameter is `graph_` plus the generator's parameter name. `help_list(what="graph_families")` lists the families with their parameters: `cycle`, `path`, `complete`, `lattice_2d` (`graph_rows`, `graph_cols`, `graph_periodic`), `watts_strogatz` and `newman_watts` (`graph_k`, `graph_beta` or `graph_p`), `erdos_renyi` (`graph_p`), `random_regular` (`graph_d`), and `file`.

- `graph_k` is the total degree K, even, with K/2 neighbours per side of the ring.
- A parameter given for a family that does not take it is an error. Defaults never are.
- **Collections.** `graph_count` G realizations, graph i drawn from seed `(graph_seed, i)`. The same graph seed gives the same collection at every measurement rate. A graph is never redrawn during a trajectory.
- **Files.** `graph=file` with `graph_file` takes any geometry. The JSON format is `{"num_nodes": n, "graphs": [[[a,b],...],...], "weights": [...]}`; a single graph may use `"edges"` instead of `"graphs"`. The NPZ format has `num_nodes`, `edges`, `ptr` and optionally `weights`. Nodes must lie in [0, n), with no self-loops or repeated edges, and `num_nodes` must equal the requested size.

Tools:

- `graph_build(num_nodes, graph..., output_file, show_edges)` saves a collection and summarises it (count, edges, degrees); `show_edges` lists edges for graphs up to 64 nodes. The saved file feeds `graph_file`.
- `graph_stats` reports, as mean and standard error over the collection, lambda_2 of the Laplacian (the algebraic connectivity), the fraction connected, the mean shortest-path length, the average clustering, the degree range, and for ring-based families the fraction of shortcut edges. lambda_2 is dense, so use n <= 4096.

Example: `graph_stats(num_nodes=64, graph="watts_strogatz", graph_k=4, graph_beta=0.1, graph_count=20, graph_seed=1)`.
