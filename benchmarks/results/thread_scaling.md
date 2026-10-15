# Thread scaling

CPU: AMD Ryzen 9 9950X3D 16-Core Processor (16 physical cores, 32 logical processors). Numba 0.68.0, `NUMBA_NUM_THREADS` = 32, threading layer `omp` (safe for concurrent calls from several Python threads). Wall time covers the run call only (the result digest is computed afterwards). Thread counts run: [1, 2, 4, 8, 16, 32]. Repeats per row are in the `runs` column; the wall time is the median of those runs after a warm-up, and runs of tens of seconds on few threads are done once. Results identical at every thread count: True. 

## W1: ensemble, Watts-Strogatz n=64 k=4 beta=0.1, purification, p=0.1, 256 layers, random_edges; code_dimension every 32 layers, erasure_first_failure(512) and uncorrectable_region_counts(2) at the end

51200 trajectories per run.

| threads | runs | wall time (s) | trajectories/s | speedup | efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 91.812 | 557.7 | 1.00 | 1.00 |
| 2 | 1 | 45.916 | 1115 | 2.00 | 1.00 |
| 4 | 1 | 23.392 | 2189 | 3.92 | 0.98 |
| 8 | 3 | 12.044 | 4251 | 7.62 | 0.95 |
| 16 | 3 | 6.703 | 7638 | 13.70 | 0.86 |
| 32 | 3 | 4.354 | 1.176e+04 | 21.08 | 0.66 |

Most threads versus 16: 32 threads take 4.354 s against 6.703 s at 16 (1.54x, faster). The 32-thread run is under the 5 s floor, so its speedup is less precise.

## W2: as W1 at n=256

9000 trajectories per run.

| threads | runs | wall time (s) | trajectories/s | speedup | efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 95.826 | 93.92 | 1.00 | 1.00 |
| 2 | 1 | 48.446 | 185.8 | 1.98 | 0.99 |
| 4 | 1 | 24.808 | 362.8 | 3.86 | 0.97 |
| 8 | 3 | 12.791 | 703.6 | 7.49 | 0.94 |
| 16 | 3 | 7.146 | 1259 | 13.41 | 0.84 |
| 32 | 3 | 4.890 | 1840 | 19.60 | 0.61 |

Most threads versus 16: 32 threads take 4.890 s against 7.146 s at 16 (1.46x, faster). The 32-thread run is under the 5 s floor, so its speedup is less precise.

## W3: purification time, cycle n=128, single reference, p=0.1, 1024 layers, stop_when_purified, code_dimension every layer

184000 trajectories per run.

| threads | runs | wall time (s) | trajectories/s | speedup | efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 41.609 | 4422 | 1.00 | 1.00 |
| 2 | 1 | 21.008 | 8759 | 1.98 | 0.99 |
| 4 | 1 | 10.981 | 1.676e+04 | 3.79 | 0.95 |
| 8 | 3 | 5.723 | 3.215e+04 | 7.27 | 0.91 |
| 16 | 3 | 3.388 | 5.431e+04 | 12.28 | 0.77 |
| 32 | 3 | 2.407 | 7.644e+04 | 17.29 | 0.54 |

Most threads versus 16: 32 threads take 2.407 s against 3.388 s at 16 (1.41x, faster). The 32-thread run is under the 5 s floor, so its speedup is less precise.

## W4: sparsegf2.sample of a noisy bit-flip repetition-code memory (distance 5, 400 rounds, X_ERROR 0.02), phases on

200000 shots per run.

| threads | runs | wall time (s) | shots/s | speedup | efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 50.834 | 3934 | 1.00 | 1.00 |
| 2 | 1 | 25.701 | 7782 | 1.98 | 0.99 |
| 4 | 1 | 12.984 | 1.54e+04 | 3.92 | 0.98 |
| 8 | 3 | 6.731 | 2.971e+04 | 7.55 | 0.94 |
| 16 | 3 | 3.877 | 5.159e+04 | 13.11 | 0.82 |
| 32 | 3 | 2.767 | 7.228e+04 | 18.37 | 0.57 |

Most threads versus 16: 32 threads take 2.767 s against 3.877 s at 16 (1.40x, faster). The 32-thread run is under the 5 s floor, so its speedup is less precise.
