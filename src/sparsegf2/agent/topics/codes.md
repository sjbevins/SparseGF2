# Codes

`code_analyze` analyses any stabilizer code. In the purification picture the system after the dynamics holds a code with stabilizer group S on n qubits and k logical qubits. Give the code in one of these ways:

- `stabilizers`: an array of Pauli strings, such as `["XXXX","ZZZZ"]` (the [[4,2,2]] code);
- `source`: a trajectory record, an ensemble run saved with `code_snapshot` (with `trajectory` and `evaluation` choosing the code), a `*.code.npz`, or a `*.state.npz` together with `reference_qubits`;
- `state_id` of an open state, with `reference_qubits`.

`properties` is an array of call strings (default `summary`). `help_list(what="code_properties")` lists them all. The main ones:

- `summary`: n, k and the number of generators.
- `lost_information(region)`: l(A), the bits of logical information lost by erasing the region A.
- `is_correctable(region)`: true exactly when l(A) = 0.
- `first_failure(samples, seed)` and `failure_by_size(samples)`: random orderings and the first-failure size D; the fraction of uncorrectable m-qubit erasures, P_C(m).
- `exhaustive_counts(max_size)`: exact counts of uncorrectable erasures of size 1..w.
- `distance_bounds(samples, max_size)`: an upper bound min D_j (an erasure smaller than the distance is always correctable) and, with exhaustive counts, the exact distance when some count is positive, otherwise the lower bound w + 1.
- `mean_first_failure` and `failure_probability(rates, delta, samples, max_size)`: mu_D, Phi(e) and the tolerance e_delta of this code.
- `stabilizer_weights`, `stabilizers`, `contiguous_distance(order)`.

`save_to` writes the code as `*.code.npz`. Use more orderings than the ensemble used to sharpen P_C(m) for one interesting code. Details of the erasure definitions are in `erasure`.
