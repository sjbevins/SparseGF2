# Performance on this CPU

CPU: AMD Ryzen 9 9950X3D 16-Core Processor (16 physical cores, 32 logical processors), Windows-11-10.0.26200-SP0. Python 3.14.7, NumPy 2.5.4, Numba 0.68.0 (`NUMBA_NUM_THREADS` = 32, threading layer `omp`), sparsegf2 3.0.0.dev12.

Topology (`sparsegf2.performance.cpu_topology`, source `windows`): 16 cores of 2 logical processors; L1d 49152 bytes and L2 1048576 bytes per core; L3: 96 MiB shared by logical processors 0-15, 32 MiB shared by logical processors 16-31.

Workload: the sweep's (Watts–Strogatz k = 4, beta = 0.1, purification picture, n // 2 random-edge gates per layer, T = 4n unless stated, `code_dimension` every layer, `tableau_snapshot` on 12 geometric checkpoints). `load` is the CPU load of other processes, in percent of all logical processors, just before the measurement.

## Where the time goes (one thread, ms per trajectory)

| n | p | full | no snapshots | gates only (p = 0) | k at the end only | load % |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.05 | 0.491 | 0.425 | 0.198 | 0.368 | 14.6 |
| 64 | 0.2 | 0.739 | 0.708 | 0.198 | 0.660 | 7.3 |
| 256 | 0.05 | 13.458 | 12.776 | 3.656 | 12.438 | 14.9 |
| 256 | 0.2 | 23.215 | 22.805 | 3.722 | 22.400 | 11.2 |
| 1024 | 0.05 | 641.810 | 628.601 | 55.276 | 626.294 | 16.6 |
| 1024 | 0.2 | 1318.966 | 1313.846 | 56.071 | 1308.869 | 10.6 |

One `MonitoredCircuit.run` call of one trajectory and one layer: 0.272 ms (the fixed cost per call).

## Threads and placement (trajectories per second)

| n | p | T | placement | threads | trajectories | trajectories/s | load % |
|---:|---:|---:|:---|---:|---:|---:|---:|
| 64 | 0.05 | 256 | all | 32 | 32768 | 3.519e+04 | 3.2 |
| 64 | 0.05 | 256 | physical | 16 | 16384 | 2.677e+04 | 6.3 |
| 64 | 0.2 | 256 | all | 1 | 2048 | 1245 | 5.7 |
| 64 | 0.2 | 256 | all | 8 | 8192 | 7762 | 4.3 |
| 64 | 0.2 | 256 | all | 16 | 16384 | 1.45e+04 | 5.2 |
| 64 | 0.2 | 256 | all | 32 | 32768 | 2.531e+04 | 4.5 |
| 64 | 0.2 | 256 | physical | 8 | 8192 | 8915 | 6.1 |
| 64 | 0.2 | 256 | physical | 16 | 16384 | 1.759e+04 | 5.0 |
| 64 | 0.2 | 256 | vcache | 8 | 8192 | 7855 | 6.9 |
| 64 | 0.2 | 256 | vcache | 16 | 16384 | 1.434e+04 | 3.0 |
| 64 | 0.2 | 256 | other_ccd | 8 | 8192 | 7413 | 6.9 |
| 64 | 0.2 | 256 | other_ccd | 16 | 16384 | 1.429e+04 | 14.7 |
| 256 | 0.05 | 1024 | all | 32 | 1024 | 1252 | 4.9 |
| 256 | 0.05 | 1024 | physical | 16 | 512 | 826.2 | 3.8 |
| 256 | 0.2 | 1024 | all | 1 | 64 | 40.72 | 7.6 |
| 256 | 0.2 | 1024 | all | 8 | 256 | 225.5 | 4.6 |
| 256 | 0.2 | 1024 | all | 16 | 512 | 430.2 | 5.0 |
| 256 | 0.2 | 1024 | all | 32 | 1024 | 758.2 | 4.0 |
| 256 | 0.2 | 1024 | physical | 8 | 256 | 308.7 | 2.1 |
| 256 | 0.2 | 1024 | physical | 16 | 512 | 541.6 | 16.8 |
| 256 | 0.2 | 1024 | vcache | 8 | 256 | 239.1 | 7.2 |
| 256 | 0.2 | 1024 | vcache | 16 | 512 | 421.1 | 7.8 |
| 256 | 0.2 | 1024 | other_ccd | 8 | 256 | 229.1 | 7.3 |
| 256 | 0.2 | 1024 | other_ccd | 16 | 512 | 434.5 | 10.0 |
| 1024 | 0.05 | 4096 | all | 32 | 64 | 23.04 | 14.3 |
| 1024 | 0.05 | 4096 | physical | 16 | 32 | 21.31 | 3.4 |
| 1024 | 0.2 | 4096 | all | 1 | 4 | 0.718 | 6.1 |
| 1024 | 0.2 | 4096 | all | 8 | 16 | 3.297 | 11.3 |
| 1024 | 0.2 | 4096 | all | 16 | 32 | 6.392 | 4.3 |
| 1024 | 0.2 | 4096 | all | 32 | 64 | 10.69 | 9.9 |
| 1024 | 0.2 | 4096 | physical | 8 | 16 | 5.29 | 9.8 |
| 1024 | 0.2 | 4096 | physical | 16 | 32 | 9.935 | 7.0 |
| 1024 | 0.2 | 4096 | vcache | 8 | 16 | 3.655 | 3.3 |
| 1024 | 0.2 | 4096 | vcache | 16 | 32 | 6.134 | 8.4 |
| 1024 | 0.2 | 4096 | other_ccd | 8 | 16 | 3.223 | 4.8 |
| 1024 | 0.2 | 4096 | other_ccd | 16 | 32 | 6.177 | 5.0 |
| 1280 | 0.2 | 256 | all | 32 | 64 | 101.3 | 7.5 |
| 1280 | 0.2 | 256 | physical | 16 | 32 | 99.34 | 7.5 |
| 1280 | 0.2 | 256 | vcache | 16 | 32 | 59.3 | 3.6 |
| 1280 | 0.2 | 256 | other_ccd | 16 | 32 | 60.35 | 6.8 |
| 1536 | 0.2 | 256 | all | 32 | 64 | 61.77 | 3.0 |
| 1536 | 0.2 | 256 | physical | 16 | 32 | 59.85 | 2.3 |
| 1536 | 0.2 | 256 | vcache | 16 | 32 | 37.15 | 7.1 |
| 1536 | 0.2 | 256 | other_ccd | 16 | 32 | 35.95 | 3.2 |
| 1792 | 0.2 | 256 | all | 32 | 64 | 40.23 | 5.7 |
| 1792 | 0.2 | 256 | physical | 16 | 32 | 43.82 | 8.3 |
| 1792 | 0.2 | 256 | vcache | 16 | 32 | 24.79 | 6.2 |
| 1792 | 0.2 | 256 | other_ccd | 16 | 32 | 20.29 | 2.0 |
| 2048 | 0.2 | 256 | all | 32 | 64 | 22.9 | 4.0 |
| 2048 | 0.2 | 256 | physical | 16 | 32 | 30.03 | 4.0 |
| 2048 | 0.2 | 256 | vcache | 16 | 32 | 17.27 | 8.4 |
| 2048 | 0.2 | 256 | other_ccd | 16 | 32 | 7.629 | 2.9 |

Common trajectories identical in every configuration: True.

## Tableau padding (all threads, p = 0.2, snapshots included)

| n | T | words per column | padding words | trajectories/s | load % |
|---:|---:|---:|---:|---:|---:|
| 512 | 2048 | 16 | 0 | 85.6 | 8.9 |
| 512 | 2048 | 16 | 1 | 85.55 | 4.7 |
| 1024 | 4096 | 32 | 0 | 5.972 | 11.1 |
| 1024 | 4096 | 32 | 1 | 9.908 | 25.4 |
| 2048 | 256 | 64 | 0 | 7.058 | 13.3 |
| 2048 | 256 | 64 | 1 | 23.23 | 14.7 |

Identical with and without padding: True.

## Gate kernel (ns per gate, random gates on a random tableau of 2n qubits)

`apply_2q` is the package's kernel (phase-free); the other columns are its linear part alone, with the 16 masks read before the word loop (as `apply_2q` now does), inside it (as it did before), and from a 16-bit table expanded in registers.

| n | words | `apply_2q` | masks before the loop | masks in the loop | 16-bit table |
|---:|---:|---:|---:|---:|---:|
| 64 | 2 | 12.56 | 8.43 | 7.17 | 8.6 |
| 256 | 8 | 34.55 | 13.84 | 32.05 | 31.26 |
| 1024 | 32 | 28.27 | 21.59 | 281.06 | 57.48 |

Identical tableaux from the three kernels: True.

## Scheduling with `stop_when_purified` (all threads)

| n | Numba chunk size | trajectories/s | final layer min / mean / max | load % |
|---:|---:|---:|---:|---:|
| 256 | 0 | 691.4 | 84 / 954.4 / 1024 | 12.2 |
| 256 | 1 | 692.9 | 84 / 954.4 / 1024 | 17.7 |
| 256 | 16 | 718.7 | 84 / 954.4 / 1024 | 14.4 |

## Small batches (ms per `run` call, median of 10)

| n | trajectories | 1 thread | 32 threads |
|---:|---:|---:|---:|
| 64 | 1 | 1.024 | 1.088 |
| 64 | 2 | 1.996 | 1.112 |
| 64 | 4 | 3.568 | 1.191 |
| 64 | 8 | 6.673 | 1.261 |
| 64 | 16 | 13.187 | 1.374 |
| 64 | 32 | 25.973 | 1.806 |

## End to end (`sweep.run` of one condition with snapshots, `sweep_throughput.measure`)

| n | p | setting | trajectories | parts | seconds | trajectories/s | engine s | load % |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|
| 64 | 0.05 | all processors, 2 writers | 16384 | 4 | 3.09 | 5304 | 0.53 | 2.0 |
| 64 | 0.05 | all processors, 4 writers | 16384 | 4 | 2.57 | 6385 | 0.55 | 7.5 |
| 64 | 0.05 | all processors, 8 writers | 16384 | 4 | 2.65 | 6182 | 0.54 | 14.9 |
| 64 | 0.05 | engine on the V-Cache CCD (16 threads), 4 writers on the other CCD | 16384 | 4 | 2.66 | 6164 | 0.82 | 5.2 |
| 64 | 0.05 | engine on the V-Cache CCD (16 threads), 8 writers on the other CCD | 16384 | 4 | 2.68 | 6120 | 0.81 | 28.6 |
| 256 | 0.05 | all processors, 2 writers | 6144 | 22 | 10.52 | 584 | 5.97 | 9.2 |
| 256 | 0.05 | all processors, 4 writers | 6144 | 22 | 7.70 | 798.4 | 6.23 | 22.1 |
| 256 | 0.05 | all processors, 8 writers | 6144 | 22 | 7.72 | 796 | 6.31 | 17.5 |
| 256 | 0.05 | engine on the V-Cache CCD (16 threads), 4 writers on the other CCD | 6144 | 21 | 9.67 | 635.5 | 8.33 | 5.5 |
| 256 | 0.05 | engine on the V-Cache CCD (16 threads), 8 writers on the other CCD | 6144 | 21 | 9.63 | 638 | 8.28 | 13.4 |
| 1024 | 0.2 | parts of 19 trajectories (the PART_BYTES cap alone) | 64 | 4 | 7.98 | 8.02 | 7.80 | 16.7 |
| 1024 | 0.2 | default chunks (whole rounds of the threads) | 64 | 2 | 6.47 | 9.89 | 6.07 | 1.8 |

Total wall time 499 s. Sections measured again in a later call: gates, placements.
