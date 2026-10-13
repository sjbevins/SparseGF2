# Example studies

Complete, runnable scripts. Each prints a summary and saves its results
(`EnsembleResult.save`, reload with `mipt.EnsembleResult.load`, or JSON).

| script | study |
|---|---|
| [`purification_time.py`](purification_time.py) | single-reference purification time on a collection of Watts–Strogatz graphs: `--graphs` realizations, each evolved with `--circuits-per-graph` random circuits (default 1; keep `--graph-seed` to reuse the same graphs at every `--p`), a matching scramble, `S(R)` every layer, stop at purification and censoring at 8n; prints the Kaplan–Meier median of tau |
| [`transition_scan.py`](transition_scan.py) | the measurement-induced transition: half-cut entropy and −I₃ across a p-sweep for several sizes, on a ring, a Watts–Strogatz graph or all-to-all |
| [`code_and_erasure.py`](code_and_erasure.py) | erasure protection of the generated code versus depth, on a graph collection in the purification picture: code rate k/n, the fraction of circuits with k > 0, first-failure sizes and exhaustive counts of uncorrectable erasures, Φ(e) for independent erasure at rate e (hybrid estimator), e_δ conditioned on k > 0, and the final code saved and analysed again as a `StabilizerCode` |

Run them from the repository root after `pip install -e "./v3"`, e.g.

```sh
python v3/examples/purification_time.py --n 64 --beta 0.1 --p 0.29 --graphs 10000
python v3/examples/purification_time.py --n 64 --p 0.25 --graphs 2500 --circuits-per-graph 4
python v3/examples/transition_scan.py --sizes 32 64 --trajectories 500
python v3/examples/code_and_erasure.py --n 64 --p 0.05 --layers 256 --every 32 --out code.npz
```

`code_and_erasure.py` also takes `--graphs`, `--circuits-per-graph`,
`--samples` (random orderings per code), `--exhaustive` (the largest
exhaustively counted erasure size w) and `--delta`.
