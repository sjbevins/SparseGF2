**Milliseconds per circuit, p = 0.05** (depth 2n; speed-up over Stim in brackets)

| n | SparseGF2 v3 | SparseGF2 v3, phase-free | Stim | Qiskit Aer (stabilizer) | Cirq (Clifford) |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.045 (3.8×) | 0.043 (3.9×) | 0.169 | 40.2 | 180 |
| 32 | 0.134 (4.4×) | 0.121 (4.9×) | 0.590 | 175 | 702 |
| 64 | 0.828 (2.2×) | 0.569 (3.2×) | 1.83 | 807 | 4,683 |
| 128 | 2.80 (4.4×) | 1.59 (7.7×) | 12.3 | 2,919 | – |
| 256 | 14.9 (4.6×) | 7.89 (8.6×) | 67.8 | 12,632 | – |
| 512 | 142 (3.2×) | 51.4 (8.7×) | 449 | – | – |
| 1024 | 1,286 (3.4×) | 430 (10.2×) | 4,379 | – | – |

**Milliseconds per circuit, p = 0.16** (depth 2n; speed-up over Stim in brackets)

| n | SparseGF2 v3 | SparseGF2 v3, phase-free | Stim | Qiskit Aer (stabilizer) | Cirq (Clifford) |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.140 (1.8×) | 0.136 (1.9×) | 0.257 | 65.7 | 182 |
| 32 | 0.258 (4.3×) | 0.166 (6.7×) | 1.11 | 196 | 816 |
| 64 | 0.951 (3.1×) | 0.660 (4.4×) | 2.90 | 789 | 5,960 |
| 128 | 7.69 (2.5×) | 2.90 (6.5×) | 18.9 | 3,243 | – |
| 256 | 39.3 (3.1×) | 14.3 (8.5×) | 122 | 12,667 | – |
| 512 | 316 (3.3×) | 89.9 (11.4×) | 1,026 | – | – |
| 1024 | 3,025 (3.4×) | 796 (12.9×) | 10,276 | – | – |

**Milliseconds per circuit, p = 0.3** (depth 2n; speed-up over Stim in brackets)

| n | SparseGF2 v3 | SparseGF2 v3, phase-free | Stim | Qiskit Aer (stabilizer) | Cirq (Clifford) |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.087 (3.6×) | 0.085 (3.7×) | 0.312 | 43.6 | 178 |
| 32 | 0.590 (1.5×) | 0.333 (2.6×) | 0.873 | 187 | 1,141 |
| 64 | 1.28 (4.4×) | 1.16 (4.9×) | 5.68 | 665 | 7,349 |
| 128 | 6.59 (6.0×) | 4.50 (8.8×) | 39.5 | 4,109 | – |
| 256 | 39.0 (4.9×) | 20.9 (9.1×) | 192 | 13,027 | – |
| 512 | 521 (3.2×) | 140 (12.0×) | 1,672 | – | – |
| 1024 | 4,598 (3.5×) | 1,103 (14.5×) | 16,010 | – | – |

**Ensemble engine throughput** (fresh random circuits, depth 2n, 4 threads), circuits per second

| p | n = 16 | n = 32 | n = 64 | n = 128 | n = 256 | n = 512 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 180,855 | 48,985 | 12,860 | 2,803 | 538 | 78 |
| 0.16 | 127,998 | 35,072 | 7,035 | 1,658 | 294 | 37 |
| 0.3 | 96,749 | 27,296 | 6,390 | 1,312 | 225 | 38 |

**Measuring the half-cut entropy (n qubits) after every layer**, milliseconds per circuit (p = 0.16, min(2n, 256) layers, one thread; Stim's observed time split into its phases; in brackets, Stim's evolve + measure time over SparseGF2's)

| n | layers | SparseGF2 evolve | SparseGF2 evolve + measure | Stim evolve | Stim evolve + measure | Stim evolve / extract / rank |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 64 | 0.123 | 0.269 | 0.762 | 1.86 (6.9×) | 0.828 / 0.786 / 0.222 |
| 64 | 128 | 0.460 | 1.10 | 3.01 | 6.08 (5.5×) | 3.28 / 1.43 / 1.29 |
| 128 | 256 | 2.39 | 6.27 | 21.3 | 37.4 (6.0×) | 22.5 / 4.53 / 10.1 |
| 256 | 256 | 6.08 | 15.5 | 70.3 | 122 (7.9×) | 74.5 / 7.48 / 40.2 |
| 512 | 256 | 19.1 | 47.2 | 308 | 508 (11×) | 301 / 20.8 / 186 |
| 1024 | 256 | 80.9 | 175 | 1,161 | 2,058 (12×) | 1,232 / 39.0 / 786 |

**Measuring the code dimension k = S(R) (2n qubits) after every layer**, milliseconds per circuit (p = 0.16, min(2n, 256) layers, one thread; Stim's observed time split into its phases; in brackets, Stim's evolve + measure time over SparseGF2's)

| n | layers | SparseGF2 evolve | SparseGF2 evolve + measure | Stim evolve | Stim evolve + measure | Stim evolve / extract / rank |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 64 | 0.139 | 0.153 | 0.999 | 1.79 (12×) | 0.932 / 0.610 / 0.213 |
| 64 | 128 | 0.590 | 0.606 | 8.00 | 11.5 (19×) | 7.68 / 2.24 / 1.49 |
| 128 | 256 | 2.96 | 2.90 | 40.2 | 57.2 (20×) | 40.1 / 6.91 / 9.90 |
| 256 | 256 | 9.30 | 10.4 | 147 | 200 (19×) | 143 / 16.9 / 39.9 |
| 512 | 256 | 51.4 | 46.4 | 610 | 887 (19×) | 638 / 37.3 / 212 |
| 1024 | 256 | 210 | 217 | 4,215 | 4,568 (21×) | 3,464 / 158 / 945 |

**Erasure screening per code**, milliseconds (one thread; median over the codes of each size; baseline = compiled from-scratch evaluation of l(A), speed-up in brackets)

| n | codes | mean k | every erasure of 1 or 2 qubits | every erasure of up to 3 qubits | first failure, 100 random orderings |
|---:|---:|---:|---:|---:|---:|
| 64 | 3 | 33 | 0.081 vs 0.240 (3.0×) | 1.15 vs 6.95 (6.0×) | 0.191 vs 0.293 (1.5×) |
| 128 | 3 | 65 | 0.329 vs 0.978 (3.0×) | 7.99 vs 54.1 (6.8×) | 0.453 vs 1.09 (2.4×) |
| 256 | 3 | 120 | 1.09 vs 4.82 (4.4×) | 36.3 vs 524 (14×) | 2.04 vs 13.4 (6.6×) |
| 512 | 3 | 249 | 4.60 vs 22.0 (4.8×) | 244 vs 5,234 (21×) | 5.52 vs 27.4 (5.0×) |
