# Sweep throughput

CPU: AMD Ryzen 9 9950X3D 16-Core Processor (16 physical cores, 32 logical processors), Windows-11-10.0.26200-SP0. Python 3.14.7, NumPy 2.5.4, Numba 0.68.0 (`NUMBA_NUM_THREADS` = 32, threading layer `omp`), sparsegf2 3.0.0.dev12.

One condition per `sweep.run` call, in a fresh directory: Watts–Strogatz graphs (k = 4, beta = 0.1), the purification picture, n // 2 random-edge gates per layer, T = 4n layers, `code_dimension` at every layer, and with `snapshots` a `tableau_snapshot` on the geometric checkpoints (count 12); default threads and chunks, four circuits per graph. `seconds` is the wall time of the whole call (graph file, index, engine, compressed parts), `engine` the engine's share; disk bytes per trajectory count the condition's part files only.

| n | p | snapshots | trajectories | seconds | trajectories/s | engine s | disk bytes/trajectory |
|---:|---:|:---|---:|---:|---:|---:|---:|
| 64 | 0.05 | no | 49152 | 4.17 | 1.18e+04 | 1.11 | 79 |
| 64 | 0.05 | yes | 24576 | 23.95 | 1026 | 0.61 | 18,875 |
| 64 | 0.2 | no | 49152 | 5.11 | 9624 | 1.84 | 96 |
| 64 | 0.2 | yes | 24576 | 20.99 | 1171 | 0.99 | 7,877 |
| 128 | 0.05 | no | 24576 | 6.02 | 4084 | 2.92 | 234 |
| 128 | 0.05 | yes | 6144 | 16.80 | 365.8 | 0.76 | 62,422 |
| 128 | 0.2 | no | 24576 | 7.94 | 3095 | 5.11 | 318 |
| 128 | 0.2 | yes | 6144 | 12.19 | 503.9 | 1.20 | 20,192 |
| 256 | 0.05 | no | 12288 | 12.17 | 1010 | 9.26 | 1,159 |
| 256 | 0.05 | yes | 3072 | 25.29 | 121.5 | 2.38 | 232,114 |
| 256 | 0.2 | no | 12288 | 21.18 | 580.2 | 17.58 | 1,550 |
| 256 | 0.2 | yes | 3072 | 17.94 | 171.2 | 4.29 | 63,792 |
| 512 | 0.05 | no | 3072 | 22.05 | 139.3 | 20.07 | 4,478 |
| 512 | 0.05 | yes | 768 | 26.07 | 29.46 | 5.11 | 867,010 |
| 512 | 0.2 | no | 3072 | 44.83 | 68.52 | 42.05 | 6,658 |
| 512 | 0.2 | yes | 768 | 19.41 | 39.57 | 9.80 | 222,618 |

Total wall time 305 s.
