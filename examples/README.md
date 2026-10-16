# Example studies

Complete, runnable scripts. Each prints a summary and saves its results
(`EnsembleResult.save`, reload with `mipt.EnsembleResult.load`, or JSON).

| script | study |
|---|---|
| [`purification_time.py`](purification_time.py) | single-reference purification time on a collection of Watts–Strogatz graphs: `--graphs` realizations, each evolved with `--circuits-per-graph` random circuits (default 1; keep `--graph-seed` to reuse the same graphs at every `--p`), a matching scramble, `S(R)` every layer, stop at purification and censoring at 8n; prints the Kaplan–Meier median of tau |
| [`transition_scan.py`](transition_scan.py) | the measurement-induced transition: half-cut entropy and −I₃ across a p-sweep for several sizes, on a ring, a Watts–Strogatz graph or all-to-all |
| [`code_and_erasure.py`](code_and_erasure.py) | erasure protection of the generated code versus depth, on a graph collection in the purification picture: code rate k/n, the fraction of circuits with k > 0, first-failure sizes and exhaustive counts of uncorrectable erasures, Φ(e) for independent erasure at rate e (hybrid estimator), e_δ conditioned on k > 0, and the final code saved and analysed again as a `StabilizerCode` |
| [`correlated_erasure.py`](correlated_erasure.py) | geometry-correlated erasures (regions drawn with weight exp(−J b(A)), b the edges leaving A) on a brickwork ring (exact samples) and a Watts–Strogatz ensemble (swap chain): P_J(m) per coupling from the in-run probe, the exact small-m values and the exact J → ∞ limit from the saved codes, Φ_J(e) and e_δ(J), and a graph-side mixing check; see [docs/observables/correlated-erasure.md](../docs/observables/correlated-erasure.md) |

Run them from the repository root after `pip install -e "."`, e.g.

```sh
python examples/purification_time.py --n 64 --beta 0.1 --p 0.29 --graphs 10000
python examples/purification_time.py --n 64 --p 0.25 --graphs 2500 --circuits-per-graph 4
python examples/transition_scan.py --sizes 32 64 --trajectories 500
python examples/code_and_erasure.py --n 64 --p 0.05 --layers 256 --every 32 --out code.npz
python examples/correlated_erasure.py --n 32 --trajectories 200 --out-dir correlated
```

`code_and_erasure.py` also takes `--graphs`, `--circuits-per-graph`,
`--samples` (random orderings per code), `--exhaustive` (the largest
exhaustively counted erasure size w) and `--delta`.
