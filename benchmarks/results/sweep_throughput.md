# Sweep throughput

CPU: AMD Ryzen 9 9950X3D 16-Core Processor (16 physical cores, 32 logical processors), Windows-11-10.0.26200-SP0. Python 3.14.7, NumPy 2.5.4, Numba 0.68.0 (`NUMBA_NUM_THREADS` = 32, threading layer `omp`), sparsegf2 3.0.0.dev12.

One condition per `sweep.run` call, in a fresh directory: Watts–Strogatz graphs (k = 4, beta = 0.1), the purification picture, n // 2 random-edge gates per layer, T = 4n layers, `code_dimension` at every layer, and with `snapshots` a `tableau_snapshot` on the geometric checkpoints (count 12); default threads, chunks, compression (`fast`) and writer threads, four circuits per graph. `seconds` is the wall time of the whole call (graph file, index, engine, saved parts), `engine` the engine's share; disk bytes per trajectory count the condition's part files only.

| n | p | snapshots | trajectories | seconds | trajectories/s | engine s | disk bytes/trajectory |
|---:|---:|:---|---:|---:|---:|---:|---:|
| 64 | 0.05 | no | 73728 | 5.43 | 1.358e+04 | 1.58 | 327 |
| 64 | 0.05 | yes | 49152 | 6.82 | 7203 | 1.50 | 25,483 |
| 64 | 0.2 | no | 73728 | 6.71 | 1.098e+04 | 2.65 | 316 |
| 64 | 0.2 | yes | 49152 | 5.76 | 8538 | 2.36 | 14,949 |
| 128 | 0.05 | no | 36864 | 7.34 | 5022 | 4.22 | 616 |
| 128 | 0.05 | yes | 12288 | 4.60 | 2670 | 1.94 | 79,071 |
| 128 | 0.2 | no | 36864 | 9.73 | 3788 | 6.71 | 603 |
| 128 | 0.2 | yes | 12288 | 4.30 | 2856 | 2.83 | 37,443 |
| 256 | 0.05 | no | 18432 | 16.33 | 1129 | 13.69 | 1,641 |
| 256 | 0.05 | yes | 6144 | 7.66 | 802.1 | 5.87 | 280,250 |
| 256 | 0.2 | no | 18432 | 27.91 | 660.5 | 25.21 | 2,119 |
| 256 | 0.2 | yes | 6144 | 10.86 | 566 | 9.63 | 109,399 |
| 512 | 0.05 | no | 4608 | 27.12 | 169.9 | 25.90 | 4,204 |
| 512 | 0.05 | yes | 1536 | 12.11 | 126.9 | 11.18 | 993,800 |
| 512 | 0.2 | no | 4608 | 57.43 | 80.24 | 56.23 | 7,128 |
| 512 | 0.2 | yes | 1536 | 22.18 | 69.24 | 21.56 | 331,521 |

Saving one part with snapshots (p = 0.05, the largest part a sweep makes) against computing it, per compression of `EnsembleResult.save`, `fsync` included (seconds, megabytes on disk):

| n | trajectories | compute s | none | fast | default |
|---:|---:|---:|---:|---:|---:|
| 128 | 1235 | 0.15 | 0.18 s, 269.8 MB | 0.74 s, 95.0 MB | 2.99 s, 77.0 MB |
| 256 | 312 | 0.27 | 0.19 s, 269.2 MB | 0.67 s, 84.9 MB | 2.21 s, 72.6 MB |

Total wall time 278 s.
