"""Generate the data behind the physics figures in the README.

Three standard experiments, each a few lines of :mod:`sparsegf2.mipt`:

1. **Measurement-induced transition.** Tripartite mutual information of the
   four quarters of a ring, ``-I3``, after ``4 n`` layers of brickwork random
   Clifford gates with Z measurements at rate ``p``. Curves for different ``n``
   cross at the critical rate (p_c ~ 0.16 for this circuit family).
2. **Purification of a single reference qubit.** A reference is Bell-paired
   with a random system qubit and scrambled in by ``2 n`` gate-only layers;
   then the monitored dynamics runs until the reference is pure. The survival
   function P(tau > t) decays on a time scale exponential in ``n`` in the mixed
   phase and logarithmic in ``n`` in the pure phase.
3. **Erasure protection of the dynamically generated code.** Starting from the
   maximally mixed state (every system qubit Bell-paired with a reference),
   the code dimension k(t) decays as the measurements purify the system; at
   ``t = n`` the probability that a uniformly random erasure of ``m`` qubits is
   correctable shows the code's erasure threshold.

Usage::

    python benchmarks/physics_data.py --out benchmarks/results/physics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from sparsegf2 import analysis, graphs, mipt  # noqa: E402


def transition(sizes, rates, trajectories):
    out = {"sizes": sizes, "rates": rates, "neg_i3": {}, "neg_i3_err": {}, "half_cut": {}}
    for n in sizes:
        rows, errs, half = [], [], []
        for p in rates:
            spec = mipt.MonitoredCircuit(
                num_system=n,
                edges=graphs.cycle(n),
                layers=4 * n,
                measurement_rate=p,
                schedule="matching",
                checkpoints=[4 * n],
                probes=[mipt.tripartite_information(), mipt.half_cut_entropy()],
            )
            res = spec.run(trajectories, seed=11)
            i3 = res.values["tripartite_information"][:, -1]
            mean, err = analysis.mean_and_error(-i3)
            rows.append(float(mean))
            errs.append(float(err))
            half.append(float(res.mean("half_cut_entropy")[-1]))
        out["neg_i3"][str(n)] = rows
        out["neg_i3_err"][str(n)] = errs
        out["half_cut"][str(n)] = half
        print(f"transition n={n} done", flush=True)
    return out


def purification(sizes, rates, trajectories, max_layers):
    out = {"sizes": sizes, "rates": rates, "layers": {}, "survival": {}}
    for p in rates:
        for n in sizes:
            spec = mipt.MonitoredCircuit(
                num_system=n,
                edges=graphs.cycle(n),
                layers=max_layers,
                measurement_rate=p,
                schedule="matching",
                reference="single",
                reference_partner="random",
                scramble_layers=2 * n,
                checkpoints="every_layer",
                stop_when_purified=True,
            )
            res = spec.run(trajectories, seed=12)
            layers, surv = res.survival()
            keep = np.unique(np.geomspace(1, len(layers), 200).astype(int) - 1)
            key = f"{p}/{n}"
            out["layers"][key] = layers[keep].tolist()
            out["survival"][key] = surv[keep].tolist()
            print(f"purification p={p} n={n}: censored {res.censored.mean():.3f}", flush=True)
    return out


def erasure(n, rates, trajectories):
    out = {
        "n": n,
        "rates": rates,
        "layers": None,
        "code_rate": {},
        "erased_fraction": None,
        "correctable": {},
    }
    checkpoints = np.unique(np.linspace(0, 2 * n, 65).astype(int)).tolist()
    for p in rates:
        spec = mipt.MonitoredCircuit(
            num_system=n,
            edges=graphs.cycle(n),
            layers=2 * n,
            measurement_rate=p,
            schedule="matching",
            reference="purification",
            checkpoints=checkpoints,
            probes=[mipt.code_dimension(), mipt.erasure_first_failure(samples=4)],
        )
        res = spec.run(trajectories, seed=13)
        out["layers"] = res.checkpoint_layers.tolist()
        out["code_rate"][str(p)] = (res.mean("code_dimension") / n).tolist()
        at = int(np.searchsorted(res.checkpoint_layers, n))
        first = res.values["erasure_first_failure"][:, at]  # (trajectories, samples)
        failing = analysis.failure_by_size(first, n).mean(axis=0)  # P(m-qubit erasure fails)
        out["erased_fraction"] = (np.arange(n + 1) / n).tolist()
        out["correctable"][str(p)] = (1.0 - failing).tolist()
        print(f"erasure p={p} done", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=str(HERE / "results" / "physics.json"))
    ap.add_argument("--quick", action="store_true", help="tiny ensembles, for a smoke test")
    args = ap.parse_args()
    scale = 0.05 if args.quick else 1.0

    def traj(k: int) -> int:
        return max(8, int(k * scale))

    t0 = time.perf_counter()
    data = {
        "transition": transition(
            [16, 32, 64, 128] if args.quick else [32, 64, 128, 256],
            [round(0.08 + 0.01 * i, 2) for i in range(17)],
            traj(800),
        ),
        "purification": purification([8, 16, 32, 64], [0.08, 0.3], traj(4000), 4000),
        "erasure": erasure(64 if args.quick else 128, [0.04, 0.08, 0.12], traj(800)),
    }
    data["seconds"] = time.perf_counter() - t0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data), encoding="utf-8")
    print(f"wrote {out} in {data['seconds']:.0f} s")


if __name__ == "__main__":
    main()
