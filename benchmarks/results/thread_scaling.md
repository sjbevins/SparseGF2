# Thread scaling

CPU: AMD Ryzen 9 9950X3D 16-Core Processor (16 physical cores, 32 logical processors). Numba 0.68.0, `NUMBA_NUM_THREADS` = 32, threading layer `omp` (safe for concurrent calls from several Python threads). Wall time is the median of 3 runs after a warm-up (one run at fewer than 8 threads, where each run lasts tens of seconds). Results were identical at every thread count.

## W1: ensemble, Watts-Strogatz n=64 k=4 beta=0.1, purification, p=0.1, 256 layers, random_edges; code_dimension every 32 layers, erasure_first_failure(512) and uncorrectable_region_counts(2) at the end

51200 trajectories per run.

| threads | runs | wall time (s) | trajectories/s | speedup | efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 92.514 | 553.4 | 1.00 | 1.00 |
| 2 | 1 | 45.744 | 1119 | 2.02 | 1.01 |
| 4 | 1 | 27.878 | 1837 | 3.32 | 0.83 |
| 8 | 3 | 14.894 | 3438 | 6.21 | 0.78 |
| 16 | 3 | 7.750 | 6606 | 11.94 | 0.75 |
| 32 | 3 | 4.561 | 1.122e+04 | 20.28 | 0.63 |

Most threads versus 16: 32 threads take 4.561 s against 7.750 s at 16 (1.70x, faster).

## W2: as W1 at n=256

9000 trajectories per run.

| threads | runs | wall time (s) | trajectories/s | speedup | efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 100.820 | 89.27 | 1.00 | 1.00 |
| 2 | 1 | 61.319 | 146.8 | 1.64 | 0.82 |
| 4 | 1 | 30.920 | 291.1 | 3.26 | 0.82 |
| 8 | 3 | 16.031 | 561.4 | 6.29 | 0.79 |
| 16 | 3 | 8.199 | 1098 | 12.30 | 0.77 |
| 32 | 3 | 4.912 | 1832 | 20.52 | 0.64 |

Most threads versus 16: 32 threads take 4.912 s against 8.199 s at 16 (1.67x, faster).

## W3: purification time, cycle n=128, single reference, p=0.1, 1024 layers, stop_when_purified, code_dimension every layer

184000 trajectories per run.

| threads | runs | wall time (s) | trajectories/s | speedup | efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 45.320 | 4060 | 1.00 | 1.00 |
| 2 | 1 | 28.538 | 6447 | 1.59 | 0.79 |
| 4 | 1 | 15.176 | 1.212e+04 | 2.99 | 0.75 |
| 8 | 3 | 8.784 | 2.095e+04 | 5.16 | 0.64 |
| 16 | 3 | 5.306 | 3.468e+04 | 8.54 | 0.53 |
| 32 | 3 | 3.493 | 5.268e+04 | 12.98 | 0.41 |

Most threads versus 16: 32 threads take 3.493 s against 5.306 s at 16 (1.52x, faster).

## W4: sparsegf2.sample of a noisy bit-flip repetition-code memory (distance 5, 400 rounds, X_ERROR 0.02), phases on

200000 shots per run.

| threads | runs | wall time (s) | shots/s | speedup | efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 55.584 | 3598 | 1.00 | 1.00 |
| 2 | 1 | 35.565 | 5623 | 1.56 | 0.78 |
| 4 | 1 | 20.053 | 9973 | 2.77 | 0.69 |
| 8 | 3 | 12.378 | 1.616e+04 | 4.49 | 0.56 |
| 16 | 3 | 8.033 | 2.49e+04 | 6.92 | 0.43 |
| 32 | 3 | 6.323 | 3.163e+04 | 8.79 | 0.27 |

Most threads versus 16: 32 threads take 6.323 s against 8.033 s at 16 (1.27x, faster).
