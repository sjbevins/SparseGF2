"""code_analyze: every erasure and structure property of a stabilizer code, on request."""

from __future__ import annotations

import difflib
from collections.abc import Callable
from typing import Any

import numpy as np

from sparsegf2 import analysis, api
from sparsegf2.agent.grammar import bind, parse_calls, plain
from sparsegf2.agent.registry import Param, ToolDef, register
from sparsegf2.agent.render import curve, fmt, lines
from sparsegf2.codes import StabilizerCode

_PAULI = np.array(["_", "X", "Z", "Y"])


def _qubits(code: StabilizerCode, value: Any, what: str) -> list[int]:
    return api._qubit_list(plain(value), what, code.num_qubits)


def _ff(code, cache, samples=512, seed=0):
    key = ("ff", int(samples), int(seed))
    if key not in cache:
        cache[key] = code.first_failure_sizes(int(samples), seed=int(seed))
    return cache[key]


def _counts(code, cache, max_size=2):
    key = ("counts", int(max_size))
    if key not in cache:
        cache[key] = code.uncorrectable_counts(int(max_size))
    return cache[key]


def _summary(code, cache):
    return {"n": code.num_qubits, "k": code.num_logical, "generators": code.num_stabilizers}


def _first_failure(code, cache, samples=512, seed=0):
    d = _ff(code, cache, samples, seed)
    return {
        "samples": int(d.size),
        "mean": float(d.mean()),
        "min": int(d.min()),
        "max": int(d.max()),
        "p_c": analysis.failure_by_size(d, code.num_qubits),
    }


def _failure_by_size(code, cache, samples=512, seed=0):
    return analysis.failure_by_size(_ff(code, cache, samples, seed), code.num_qubits)


def _exhaustive(code, cache, max_size=2):
    c = _counts(code, cache, max_size)
    return {"counts": c, "p_c_exact": analysis.exact_failure_by_size(c, code.num_qubits)}


def _bounds(code, cache, samples=512, max_size=2, seed=0):
    low, high = analysis.distance_bounds(
        _ff(code, cache, samples, seed),
        code.num_qubits,
        exact_counts=_counts(code, cache, max_size),
    )
    return {"low": int(low), "high": int(high)}


def _mu(code, cache, samples=512, max_size=2, seed=0):
    return float(
        analysis.mean_first_failure(
            _ff(code, cache, samples, seed),
            code.num_qubits,
            exact_counts=_counts(code, cache, max_size),
        )
    )


def _phi(code, cache, rates=(0.01, 0.02, 0.05, 0.1), delta=0.01, samples=512, max_size=2, seed=0):
    d = _ff(code, cache, samples, seed)
    counts = _counts(code, cache, max_size)
    rates = [float(r) for r in (rates if isinstance(rates, list | tuple) else [rates])]
    phi = analysis.erasure_failure_probability(d, code.num_qubits, rates, exact_counts=counts)
    e_d = analysis.erasure_tolerance(
        d[None, :], code.num_qubits, float(delta), exact_counts=counts[None, :]
    )
    return {
        "rates": rates,
        "phi": np.atleast_1d(phi).tolist(),
        "delta": float(delta),
        "e_delta": e_d,
    }


def _weights(code, cache):
    m = code.stabilizer_matrix()
    n = code.num_qubits
    w = ((m[:, :n] | m[:, n:]) > 0).sum(axis=1) if len(m) else np.zeros(0, int)
    return {
        "mean": float(w.mean()) if w.size else 0.0,
        "min": int(w.min()) if w.size else 0,
        "max": int(w.max()) if w.size else 0,
    }


def _stabilizers(code, cache):
    n = code.num_qubits
    if n > api.MAX_STABILIZER_QUBITS:
        raise ValueError(
            f"stabilizers are listed only up to {api.MAX_STABILIZER_QUBITS} qubits; "
            "use stabilizer_weights"
        )
    m = code.stabilizer_matrix().astype(int)
    return ["".join(_PAULI[m[i, :n] + 2 * m[i, n:]]) for i in range(len(m))]


def _contiguous(code, cache, order=None, periodic=True):
    seq = None if order is None else _qubits(code, order, "contiguous_distance.order")
    return code.contiguous_distance(seq, periodic=bool(periodic))


def _lost(code, cache, region):
    return code.lost_information(_qubits(code, region, "lost_information.region"))


def _correctable(code, cache, region):
    return code.is_correctable(_qubits(code, region, "is_correctable.region"))


CODE_PROPERTIES: dict[str, tuple[tuple[str, ...], str, Callable]] = {
    "summary": ((), "n, k (logical qubits) and the number of generators", _summary),
    "lost_information": (("region",), "bits of logical information lost by erasing region", _lost),
    "is_correctable": (("region",), "whether erasing region is correctable", _correctable),
    "first_failure": (
        ("samples", "seed"),
        "first-failure sizes D over random orderings: mean, range and P_C(m)",
        _first_failure,
    ),
    "failure_by_size": (("samples", "seed"), "P_C(m), m = 0..n, from orderings", _failure_by_size),
    "exhaustive_counts": (
        ("max_size",),
        "uncorrectable m-qubit erasures for m <= max_size, and the exact P_C(m)",
        _exhaustive,
    ),
    "distance_bounds": (
        ("samples", "max_size", "seed"),
        "bounds on the distance d (exact when d <= max_size)",
        _bounds,
    ),
    "mean_first_failure": (("samples", "max_size", "seed"), "mu_D = E[D] (hybrid estimator)", _mu),
    "failure_probability": (
        ("rates", "delta", "samples", "max_size", "seed"),
        "Phi(e) at each erasure rate and e_delta",
        _phi,
    ),
    "stabilizer_weights": ((), "generator weights: mean, min, max (basis dependent)", _weights),
    "stabilizers": ((), "the generators as Pauli strings (up to 32 qubits)", _stabilizers),
    "contiguous_distance": (
        ("order", "periodic"),
        "shortest contiguous window (in order) whose erasure is uncorrectable",
        _contiguous,
    ),
}


def load_code(ctx, a: dict[str, Any]) -> tuple[StabilizerCode, str]:
    given = [k for k in ("stabilizers", "source", "state_id") if a.get(k) is not None]
    if len(given) != 1:
        raise ValueError("give exactly one of stabilizers, source or state_id")
    if given == ["stabilizers"]:
        return StabilizerCode(list(a["stabilizers"])), "the given stabilizers"
    if given == ["state_id"]:
        if not a.get("reference_qubits"):
            raise ValueError("state_id needs reference_qubits (e.g. 8:16)")
        sim = ctx.states.get(a["state_id"])
        ref = api._qubit_list(a["reference_qubits"], "reference_qubits", sim.num_qubits)
        return StabilizerCode.from_simulator(sim, ref), f"state {a['state_id']}"
    path = ctx.path(a["source"], what="source", must_exist=True)
    if path.name.endswith(".code.npz"):
        with np.load(path, allow_pickle=False) as z:
            return StabilizerCode(z["stabilizers"], num_qubits=int(z["num_qubits"])), ctx.show(path)
    if path.name.endswith(".state.npz"):
        if not a.get("reference_qubits"):
            raise ValueError("a state file needs reference_qubits (e.g. 8:16)")
        from sparsegf2.simulator import Simulator

        sim = Simulator.load(path)
        ref = api._qubit_list(a["reference_qubits"], "reference_qubits", sim.num_qubits)
        return StabilizerCode.from_simulator(sim, ref), ctx.show(path)
    res = api._load_directory(path)[0] if path.is_dir() else api._load_result_file(path)
    if res is None:
        raise ValueError(f"no saved result in {ctx.show(path)}")
    evaluations = len(res.layers(next(iter(res.snapshots))) if res.snapshots else [])
    evaluation = a.get("evaluation", -1)
    if evaluation < 0:
        evaluation += evaluations
    t = a.get("trajectory")
    numbers = np.asarray(res.trajectories)
    if t is None:
        row = 0
    else:
        hits = np.flatnonzero(numbers == t)
        if not hits.size:
            raise ValueError(
                f"trajectory {t} is not in {ctx.show(path)} (it holds "
                f"{numbers.min()}..{numbers.max()})"
            )
        row = int(hits[0])
    code = res.code(row, evaluation)
    return code, f"{ctx.show(path)}, trajectory {int(numbers[row])}, evaluation {evaluation}"


def _analyze(ctx, a):
    code, origin = load_code(ctx, a)
    cache: dict = {}
    results = []
    for i, call in enumerate(parse_calls(a.get("properties") or ["summary"], what="properties")):
        if call.name not in CODE_PROPERTIES:
            close = difflib.get_close_matches(call.name, list(CODE_PROPERTIES), n=2)
            hint = f"; did you mean {', '.join(close)}?" if close else ""
            raise ValueError(
                f"properties[{i}]: unknown property {call.name!r}{hint} "
                "(help_list(what=code_properties))"
            )
        names, _, fn = CODE_PROPERTIES[call.name]
        kwargs = plain(bind(call, names, what=f"properties[{i}]"))
        results.append(
            {"property": call.text.strip(), "name": call.name, "value": fn(code, cache, **kwargs)}
        )
    out = {"origin": origin, "n": code.num_qubits, "k": code.num_logical, "results": results}
    if a.get("save_to"):
        path = ctx.path(a["save_to"], what="save_to")
        if not path.name.endswith(".code.npz"):
            raise ValueError("code files end in .code.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            np.savez(
                fh,
                stabilizers=code.stabilizer_matrix().astype(np.uint8),
                num_qubits=np.int64(code.num_qubits),
            )
        out["saved"] = ctx.show(path)
    return out


def _show(name: str, label: str, v: Any) -> str:
    if name == "summary":
        return f"{label}: n={v['n']}, k={v['k']}, generators={v['generators']}"
    if name == "first_failure":
        return (
            f"{label}: mean D={fmt(v['mean'])} over {v['samples']} orderings, "
            f"range {v['min']}..{v['max']}; P_C(m) " + curve(v["p_c"])
        )
    if name == "failure_by_size":
        return f"{label}: P_C(m) " + curve(v)
    if name == "exhaustive_counts":
        return (
            f"{label}: counts {' '.join(fmt(x) for x in v['counts'])}; exact P_C(m), m=1..: "
            + " ".join(fmt(x, 3) for x in v["p_c_exact"])
        )
    if name == "distance_bounds":
        if v["low"] == v["high"]:
            return f"{label}: d = {v['low']}"
        return f"{label}: {v['low']} <= d <= {v['high']}"
    if name == "failure_probability":
        pairs = ", ".join(
            f"Phi({fmt(e)})={fmt(p)}" for e, p in zip(v["rates"], v["phi"], strict=True)
        )
        return f"{label}: {pairs}; e_{fmt(v['delta'])}={fmt(v['e_delta'])}"
    if name == "stabilizer_weights":
        return f"{label}: mean {fmt(v['mean'])}, min {v['min']}, max {v['max']}"
    if name == "stabilizers":
        return f"{label}: " + " ".join(v)
    return f"{label}: {fmt(v)}"


def _render(d, ctx):
    head = f"code_analyze: [[{d['n']}, {d['k']}]] code from {d['origin']}"
    body = [_show(r["name"], r["property"], r["value"]) for r in d["results"]]
    tail = f"saved: {d['saved']}" if d.get("saved") else ""
    return lines(head, *body, tail)


register(
    ToolDef(
        name="code_analyze",
        stack="code",
        summary="Any property of a stabilizer code, on request.",
        description=(
            "Analyse a stabilizer code: given as Pauli strings (stabilizers), saved (source: a "
            "traj_run record, an ensemble run with code snapshots, a .code.npz or a .state.npz with "
            "reference_qubits), or an open state (state_id with reference_qubits). properties are "
            "call strings: "
            "summary, first_failure(samples=512), exhaustive_counts(3), distance_bounds, "
            "mean_first_failure, failure_probability(rates=[0.01,0.05], delta=0.01), "
            "lost_information(0:4), is_correctable([0,1]), stabilizer_weights, stabilizers, "
            'contiguous_distance (help_list(what=code_properties)). Example: {"stabilizers": '
            '["XXXX", "ZZZZ"], "properties": ["summary", "distance_bounds"]}'
        ),
        params=(
            Param(
                "stabilizers",
                "array",
                'Generators as Pauli strings, e.g. ["XXXX", "ZZZZ"].',
                items="string",
            ),
            Param(
                "source", "string", "A saved code, trajectory record, ensemble run or state file."
            ),
            Param("state_id", "string", "An open state (with reference)."),
            Param("reference_qubits", "string", "The reference qubits of a state, e.g. 8:16."),
            Param(
                "trajectory",
                "integer",
                "Trajectory number in a saved ensemble (default: its first).",
                minimum=0,
            ),
            Param("evaluation", "integer", "Snapshot index (-1 = the last).", default=-1),
            Param("properties", "array", "Property calls (default summary).", items="string"),
            Param("save_to", "string", "Also save the code as a .code.npz file."),
        ),
        handler=_analyze,
        render=_render,
        long_running=True,
        examples=({"stabilizers": ["XXXX", "ZZZZ"], "properties": ["summary", "distance_bounds"]},),
    )
)
