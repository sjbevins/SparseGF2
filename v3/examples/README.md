# Example studies

Complete, runnable scripts. Each prints a summary and saves its results
(`EnsembleResult.save`, reload with `mipt.EnsembleResult.load`, or JSON).

| script | study |
|---|---|
| [`purification_time.py`](purification_time.py) | single-reference purification time on Watts–Strogatz graphs, the PRL production protocol (one graph per trajectory, matching scramble, `S(R)` every layer, censoring at 8n) |
| [`transition_scan.py`](transition_scan.py) | the measurement-induced transition: half-cut entropy and −I₃ across a p-sweep for several sizes, on a ring, WS graphs or all-to-all |
| [`code_and_erasure.py`](code_and_erasure.py) | the dynamically generated code vs depth: code dimension, contiguous distance, erasure correctability and i.i.d.-erasure failure probability |

Run them from the repository root after `pip install -e "./v3"`, e.g.

```sh
python v3/examples/purification_time.py --n 64 --beta 0.1 --p 0.29 --trajectories 10000
python v3/examples/transition_scan.py --sizes 32 64 --trajectories 500
python v3/examples/code_and_erasure.py --n 64 --p 0.08 --layers 128 --every 16
```
