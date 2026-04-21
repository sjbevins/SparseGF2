# SparseGF2

**A sparse stabilizer simulator that exploits the measurement-induced phase transition for asymptotic speedup**


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/) [![Status](https://img.shields.io/badge/status-beta-orange.svg)](CHANGELOG.md)

SparseGF2 is a Clifford circuit simulator built from scratch to exploit the *sparsity phase transition* in stabilizer generators. In the area-law phase of the measurement-induced phase transition (MIPT), stabilizer generators have $\mathcal{O}(1)$ support independent of system size. SparseGF2 exploits this structure with a hybrid sparse-index data structure that achieves **$\mathcal{O}(n)$ per-layer cost** in the area-law phase, versus the $\mathcal{O}(n^2)$ per-layer cost of standard tableau simulators.

## Benchmarks

### Speedup over Stim (n=512, nearest-neighbor brickwork, 8n depth)

Benchmarked with the **complete 11,520-element two-qubit Clifford group** (deterministically enumerated via `stim.Tableau.iter_all(2)`) and **RREF-verified** at every data point — the stabilizer *group* is identical to Stim's (not just the rank k).

| Measurement rate $p$ | Phase | SparseGF2 | Stim | Speedup |
|:---:|:---:|:---:|:---:|:---:|
| 0.02 | Volume-law | 29.1 s | 18.1 s | 0.6× (slower) |
| 0.10 | Volume-law | 16.5 s | 45.8 s | **2.8×** |
| 0.14 | Near-critical | 6.76 s | 58.7 s | **8.7×** |
| 0.16 | Critical | 3.27 s | 64.7 s | **19.8×** |
| 0.20 | Area-law | 2.49 s | 76.0 s | **30.6×** |
| 0.30 | Area-law | 2.56 s | 101.8 s | **39.8×** |
| 0.40 | Area-law | 2.73 s | 126.6 s | **46.5×** |

The crossover occurs near the critical measurement rate $p_c$ ≈ 0.16, which is exactly the MIPT critical point. Below $p_c$ (volume-law phase), generators are dense and Stim's bit-packed SIMD operations are faster. Above $p_c$ (area-law phase), generators become $\mathcal{O}(1)$-sparse and SparseGF2's inverted-index approach achieves an asymptotic advantage that grows with system size.

### Scaling exponents (nearest-neighbor model)

| Measurement rate $p$ | Phase | SparseGF2 $\gamma$ | Stim $\gamma$ | Theory |
|:---:|:---:|:---:|:---:|:---:|
| 0.04 | Volume-law | 2.94 | ~3.0 | $\mathcal{O}(n^3)$ |
| 0.10 | Volume-law | 2.88 | ~3.0 | O(n³) |
| 0.20 | Area-law | **1.98** | ~3.0 | $\mathcal{O}(n^2)$ |
| 0.30 | Area-law | **1.97** | ~3.0 | $\mathcal{O}(n^2)$ |

In the area-law phase, total circuit cost scales as $n^2$ ($\gamma \approx 2.0$), matching the information-theoretic lower bound for any tableau-based simulator. Standard simulators like Stim always scale as $n^3$ regardless of the measurement rate.

**Circuit parameters:** Random 2-qubit Cliffords drawn uniformly from the complete 11,520-element two-qubit Clifford group. Nearest-neighbor brickwork on the cycle graph C_n with periodic boundary conditions. Depth = 8n layers. Each gate endpoint qubit measured in the Z basis with probability *p*. Purification picture: n system qubits + n reference qubits initialized as Bell pairs. 2 timing seeds per data point, RREF verification against Stim at every (n, p) point.

---

## Theory: Why we see the speedup

### The Measurement-Induced Phase Transition

Random Clifford circuits with interleaved measurements exhibit a sharp phase transition at a critical measurement rate $p_c$:

- **Volume-law phase** ($p < p_c$): Entanglement entropy $S(A)$ scales as the volume of subsystem $A$. The quantum state retains extensive entanglement despite measurements.
- **Area-law phase** ($p>p_c$): Entanglement entropy scales as the boundary of $A$. Measurements destroy long-range entanglement.

For nearest-neighbor circuits on the cycle graph, $p_c \approx 0.16$, and the transition is in the $2D$ bond percolation universality class with correlation length exponent $\nu = 4/3$.



### The Sparsity Phase Transition in Stabilizer Generators

The MIPT has a direct manifestation in the structure of stabilizer generators. Define the **average active count** *ā* as the mean number of generators with nontrivial Pauli support on a given qubit. Empirically:

- **Volume-law** ($p < p_c$): $\bar{a} \sim n^\alpha$ with $\alpha \approx 0.9$ — generators have extensive support spanning most of the system.
- **Area-law** ($p > p_c$): $\bar{a} = \mathcal{O}(1)$ — generators are localized, each touching only a constant number of qubits independent of system size.

The transition in $\bar{a}$ occurs at the same critical point $p_c$ as the entanglement transition, because both are controlled by the same correlation length divergence.


### From Sparsity to Speedup

Standard tableau simulators (CHP, Stim, AG algorithm) process every gate by scanning **all $2n$ generators**, regardless of how many actually have support on the gate qubits. The per-gate cost is $\Theta(n)$, giving per-layer cost $\Theta(n^2)$ and total circuit cost $\Theta(n^3)$.


SparseGF2 maintains an **inverted index**: for each qubit $q$, a list of generators with nontrivial Pauli support on $q$. A gate on $(q_i, q_j)$ touches only the generators in `inv[q_i] ∪ inv[q_j]`, which has expected size $\bar{a}$. The per-gate cost is $\Theta(\bar{a})$, giving:

| | Per gate | Per layer | Total (depth D) |
|---|---|---|---|
| **AG/Stim** | $\Theta(n)$ | $\Theta(n^2)$ | $\Theta(n^2D)$ |
| **SparseGF2** | $\Theta(\bar{a})$ | $\Theta(n\cdot \bar{a})$ | $\Theta(n\cdot \bar{a}\cdot D)$ |


In the area-law phase where $\bar{a}= \mathcal{O}(1)$: SparseGF2 achieves $\Theta(n)$ per layer and $\Theta(n^2)$ total for depth $D = \Theta(n) — an **$\Theta(n)$ asymptotic speedup** over standard simulators.

