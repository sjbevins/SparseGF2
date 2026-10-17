# Costs of the study's tableau store

Machine: AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD, Windows-11-10.0.26200-SP0; 32 threads. Made by `benchmarks/study_store_costs.py`.

## Ingestion (64 trajectories, one unit)

| n | T* bootstrap | engine (s) | rest of the unit (s) | ratio | trajectories/s | snapshots/s | joint bytes/snapshot | physical bytes | file bytes/snapshot |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 200 | 0.005 | 0.50 | 109 | 127 | 1524 | 389 | 93 | 2053 |
| 32 | 0 | 0.003 | 0.24 | 68 | 270 | 3237 | 389 | 93 | 2053 |
| 64 | 200 | 0.007 | 0.57 | 76 | 113 | 1357 | 1059 | 202 | 3248 |
| 64 | 0 | 0.011 | 0.29 | 26 | 219 | 2629 | 1059 | 202 | 3253 |
| 128 | 200 | 0.038 | 0.71 | 19 | 91 | 1178 | 3180 | 486 | 5982 |
| 128 | 0 | 0.025 | 0.45 | 18 | 142 | 1840 | 3180 | 486 | 5967 |
| 256 | 200 | 0.121 | 1.50 | 12 | 43 | 555 | 11056 | 1328 | 14838 |
| 256 | 0 | 0.110 | 1.19 | 11 | 54 | 697 | 11056 | 1328 | 14828 |

## Memory of one unit (32 trajectories)

| n | estimate (MiB) | traced peak (MiB) |
|---|---|---|
| 32 | 7.9 | 3.3 |
| 64 | 12.3 | 4.9 |
| 128 | 28.7 | 12.7 |
| 256 | 83.3 | 45.8 |

## Lookup by key (n = 6)

| snapshots | trajectories | `store.tableau` (us) | `store.snapshot` (us) |
|---|---|---|---|
| 1000 | 200 | 30 | 9 |
| 10000 | 2000 | 28 | 10 |
| 100000 | 20000 | 34 | 15 |
