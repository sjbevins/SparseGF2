"""The MCP server, driven by an in-process MCP client (skipped if the mcp extra is missing)."""

import asyncio
import json
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")

from sparsegf2 import api, mcp_server  # noqa: E402

SPEC = {
    "num_system": 8,
    "graph": {"family": "cycle"},
    "layers": 16,
    "measurement_rate": 0.2,
    "reference": "single",
    "checkpoints": 4,
    "probes": [{"kind": "code_dimension"}],
    "stop_when_purified": True,
}

EXPECTED_TOOLS = {
    "capabilities",
    "inspect_circuit",
    "run_circuit",
    "run_ensemble",
    "estimate_ensemble",
    "benchmark_ensemble",
    "start_ensemble_job",
    "job_status",
    "cancel_job",
    "list_jobs",
    "load_results",
    "sample_erasure_regions",
    "erasure_curves",
}


def session(fn):
    """Run ``await fn(client)`` against a fresh in-process server."""

    async def runner():
        async with mcp.Client(mcp_server.build_server()) as client:
            return await fn(client)

    return asyncio.run(runner())


def call(name: str, arguments: dict):
    return session(lambda c: c.call_tool(name, arguments))


def ok(result) -> dict:
    assert not result.is_error, result.content[0].text
    # The text block is the same result as compact JSON (what the model reads).
    assert result.content[0].text == json.dumps(result.structured_content, separators=(",", ":"))
    return result.structured_content


def error_text(result) -> str:
    assert result.is_error
    return result.content[0].text


def test_tools_are_listed_with_good_descriptions():
    tools = session(lambda c: c.list_tools()).tools
    assert {t.name for t in tools} == EXPECTED_TOOLS
    for tool in tools:
        assert 50 < len(tool.description) < 2000, tool.name
        assert "Example:" in tool.description, tool.name
    run = next(t for t in tools if t.name == "run_circuit")
    assert run.input_schema["required"] == ["circuit"]
    assert "tick_stride" in run.input_schema["properties"]
    ensemble = next(t for t in tools if t.name == "run_ensemble")
    for kind in api.capabilities("probes")["probes"]:
        assert kind in ensemble.description  # generated from the library, never stale
    assert all(t.output_schema and t.output_schema["type"] == "object" for t in tools)


def test_the_erasure_tools_are_read_only_and_say_what_they_return():
    tools = {t.name: t for t in session(lambda c: c.list_tools()).tools}
    for name in ("sample_erasure_regions", "erasure_curves"):
        tool = tools[name]
        assert tool.annotations.read_only_hint is True, name  # they write nothing
        assert tool.annotations.open_world_hint is False, name
        assert 1000 < len(tool.description) < 2000, name  # a full description, within the limit
    regions = tools["sample_erasure_regions"]
    assert regions.input_schema["required"] == ["graph", "size", "couplings"]
    for field in ("samples", "num_nodes", "burn_in", "thin", "tempering", "chains", "seed",
                  "method", "ladder", "max_step"):  # fmt: skip
        assert field in regions.input_schema["properties"], field
    assert (
        "max_seconds" not in regions.input_schema["properties"]
    )  # a call must fit a client's wait
    for word in ("mean_boundary", "exact_source", "mixing", "nonincreasing_in_J", "[seed, i]",
                 "r_hat_limit", "z_critical", "ess_boundary"):  # fmt: skip
        assert word in regions.description, word
    curves = tools["erasure_curves"]
    assert curves.input_schema["required"] == ["path"]
    for word in ("Phi(e)", "e_delta", "of_mean_phi", "of_mean_phi_range", "interpolated_mass",
                 "truncated", "headline"):  # fmt: skip
        assert word in curves.description, word
    # the run_ensemble description keeps its room
    assert len(tools["run_ensemble"].description) < 2000


def test_results_fit_the_size_budget_as_sent():
    result = call("capabilities", {})
    assert len(result.content[0].text) <= api.MAX_OUTPUT_CHARS


def test_server_metadata():
    async def check(client):
        assert client.server_info.name == mcp_server.SERVER_NAME
        assert "quickstart" in (client.instructions or "")

    session(check)


def test_capabilities_tool():
    assert ok(call("capabilities", {})) == api.capabilities()
    assert list(ok(call("capabilities", {"section": "observables"}))) == ["observables"]
    assert "unknown section" in error_text(call("capabilities", {"section": "x"}))


def test_inspect_and_run_circuit():
    info = ok(call("inspect_circuit", {"circuit": "H 0\nCX 0 1\nM 0 1"}))
    assert info["valid"] and info["num_qubits"] == 2
    bad = ok(call("inspect_circuit", {"circuit": "H 0\nXYZZY 1"}))
    assert bad["errors"] == [{"line": 2, "message": bad["errors"][0]["message"]}]

    args = {"circuit": "H 0\nCX 0 1\nTICK\nM 0", "seed": 3,
            "observables": [{"kind": "entropy", "region": [1]}], "observe_ticks": True}  # fmt: skip
    out = ok(call("run_circuit", args))
    expected = api.run_circuit(**args)
    assert out["measurements"] == expected["measurements"]
    assert out["tick_observables"] == expected["tick_observables"]
    assert out["observables"] == {"entropy": 0}


def test_run_circuit_errors_are_actionable():
    text = error_text(call("run_circuit", {"circuit": "H 0\nXYZZY 1"}))
    assert "line 2" in text and "XYZZY" in text
    text = error_text(call("run_circuit", {"circuit": "H 0", "observables": [{"kind": "vol"}]}))
    assert "unknown observable kind" in text and "entropy" in text
    # Schema violations are reported by the SDK as tool errors too.
    assert error_text(call("run_circuit", {"circuit": "H 0", "shots": "many"}))


def test_run_ensemble_matches_api_and_guard():
    out = ok(call("run_ensemble", {"spec": SPEC, "trajectories": 20, "seed": 4}))
    expected = api.run_ensemble(SPEC, 20, seed=4)
    assert out["purification"] == expected["purification"]
    assert out["probes"] == expected["probes"]

    big = dict(SPEC, num_system=1024, layers=8192, checkpoints="final")
    text = error_text(call("run_ensemble", {"spec": big, "trajectories": 100_000}))
    assert "start_ensemble_job" in text
    text = error_text(call("run_ensemble", {"spec": dict(SPEC, layrs=3), "trajectories": 2}))
    assert "unknown key" in text and "layers" in text
    text = error_text(call("run_ensemble", {"spec": SPEC, "trajectories": 2, "max_seconds": -1}))
    assert "max_seconds must be positive" in text


def test_run_ensemble_builds_the_spec_once(monkeypatch):
    # The guard used to call estimate_ensemble and then run_ensemble, generating
    # a graph bank twice; with realizations=G that doubled the wall time.
    calls = []
    real = api._build_spec

    def counting(spec):
        calls.append(spec)
        return real(spec)

    monkeypatch.setattr(api, "_build_spec", counting)
    bank = dict(SPEC, graph={"family": "watts_strogatz", "k": 4, "beta": 0.2, "seed": 1,
                             "realizations": 5}, circuits_per_graph=2)  # fmt: skip
    out = ok(call("run_ensemble", {"spec": bank, "trajectories": 10, "seed": 2}))
    assert out["num_trajectories"] == 10 and len(calls) == 1


def test_estimate_tool():
    est = ok(call("estimate_ensemble", {"spec": SPEC, "trajectories": 10}))
    assert est["fits_sync_limit"]
    assert "by_threads" not in est and est["layout"]["files"][0] == "result.npz"
    with_threads = ok(
        call("estimate_ensemble", {"spec": SPEC, "trajectories": 10, "threads": [1, 4]})
    )
    assert [row["threads"] for row in with_threads["by_threads"]] == [1, 4]
    assert with_threads == api.estimate_ensemble(SPEC, 10, threads=[1, 4])
    text = error_text(call("estimate_ensemble", {"spec": SPEC, "trajectories": 10, "threads": [0]}))
    assert "threads" in text


def test_benchmark_tool(tmp_path):
    tools = session(lambda c: c.list_tools()).tools
    (tool,) = [t for t in tools if t.name == "benchmark_ensemble"]
    assert tool.annotations.read_only_hint  # it writes nothing
    args = {"spec": SPEC, "trajectories": 20, "threads": [1, 64], "seconds": 1}
    out = ok(call("benchmark_ensemble", args))
    assert out["trajectories"] == 20
    assert [row["threads"] for row in out["projected"]] == [1, 64]
    assert out["storage"]["disk_megabytes"] > 0
    # a call takes up to about twice `seconds`: more than half the synchronous limit is refused
    text = error_text(call("benchmark_ensemble", {**args, "seconds": 31}))
    assert "seconds" in text and "at most 30" in text
    assert "seconds" in error_text(call("benchmark_ensemble", {**args, "seconds": 0}))
    assert "threads" in error_text(call("benchmark_ensemble", {**args, "threads": [0]}))


REGIONS_ARGS = {
    "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.2, "seed": 3},
    "num_nodes": 20, "size": 5, "couplings": [0, 1.5], "samples": 40, "burn_in": 30, "seed": 6,
}  # fmt: skip


def _without_timing(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in ("elapsed_seconds", "estimated_seconds")}


def test_sample_erasure_regions_tool_matches_the_api():
    out = ok(call("sample_erasure_regions", REGIONS_ARGS))
    args = dict(REGIONS_ARGS)
    expected = api.sample_erasure_regions(
        args.pop("graph"), args.pop("size"), args.pop("couplings"), args.pop("samples"), **args
    )
    assert _without_timing(out) == _without_timing(json.loads(json.dumps(expected)))
    assert out["seed"] == 6 and [r["coupling"] for r in out["couplings"]] == [0, 1.5]
    # a lone number is a list of one coupling, and the graph may be edges
    ring = [[i, (i + 1) % 8] for i in range(8)]
    one = ok(call("sample_erasure_regions",
                  {"graph": {"edges": ring, "num_nodes": 8}, "size": 3, "couplings": 1.0,
                   "samples": 20, "seed": 1}))  # fmt: skip
    assert one["graph"]["geometry"] == "cycle" and len(one["couplings"]) == 1


def test_sample_erasure_regions_errors_are_actionable():
    def err(**change):
        return error_text(call("sample_erasure_regions", {**REGIONS_ARGS, **change}))

    assert "size must be between 1 and 19" in err(size=20)
    assert "couplings must be distinct" in err(couplings=[1, 1])
    assert "samples must be at least 8" in err(samples=3)
    assert "unknown graph family" in err(graph={"family": "moebius"})
    assert "estimated at about" in err(  # too costly: refused before anything runs
        graph={"family": "watts_strogatz", "k": 4, "beta": 0.1, "seed": 1}, num_nodes=2000,
        size=500, couplings=[0, 1, 2], samples=2000, chains=8, burn_in=None,
    )  # fmt: skip
    # schema violations: the SDK reports them as tool errors too
    for bad in ({"samples": True}, {"chains": "4"}, {"tempering": 1}, {"seed": 1.5},
                {"couplings": ["a"]}, {"max_step": True}):  # fmt: skip
        assert err(**bad), bad
    assert "max_seconds" not in err(samples=3)  # not an argument of the tool


@pytest.fixture(scope="module")
def curves_run(tmp_path_factory):
    spec = {**SPEC, "num_system": 12, "reference": "purification", "checkpoints": [8, 16],
            "stop_when_purified": False, "measurement_rate": 0.1, "schedule": "matching",
            "probes": [{"kind": "code_dimension"},
                       {"kind": "erasure_first_failure", "samples": 80},
                       {"kind": "correlated_erasure", "couplings": [0, 1], "samples": 40}]}  # fmt: skip
    out = tmp_path_factory.mktemp("mcp_curves") / "run"
    api.run_ensemble(spec, 20, seed=2, output_dir=str(out))
    return out


def test_erasure_curves_tool_matches_the_api(curves_run):
    args = {"path": str(curves_run), "e": [0.05, 0.2], "delta": 0.05, "layer": 8}
    out = ok(call("erasure_curves", args))
    expected = api.erasure_curves(str(curves_run), [0.05, 0.2], 0.05, 8)
    assert _without_timing(out) == _without_timing(json.loads(json.dumps(expected)))
    assert out["uniform"]["erasure_first_failure"]["layer"] == 8
    default = ok(call("erasure_curves", {"path": str(curves_run)}))
    assert default["layer"] is None and default["uniform"]["erasure_first_failure"]["layer"] == 16
    assert ok(call("erasure_curves", {"path": str(curves_run), "e": 0.1}))["e"] == [0.1]


def test_erasure_curves_errors_are_actionable(curves_run, tmp_path):
    assert "no such file or directory" in error_text(
        call("erasure_curves", {"path": str(tmp_path / "nowhere")})
    )
    text = error_text(call("erasure_curves", {"path": str(curves_run), "layer": 7}))
    assert "layer 7 was not evaluated" in text
    assert "delta must be in [0, 1]" in error_text(
        call("erasure_curves", {"path": str(curves_run), "delta": 2})
    )
    for bad in ({"layer": True}, {"layer": "8"}, {"delta": "x"}, {"e": "all"}, {"e": [True]}):
        assert error_text(call("erasure_curves", {"path": str(curves_run), **bad})), bad
    api.run_ensemble(SPEC, 3, seed=1, output_dir=str(tmp_path / "plain"))
    text = error_text(call("erasure_curves", {"path": str(tmp_path / "plain")}))
    assert "no erasure_first_failure or correlated_erasure probe" in text


def test_the_erasure_tools_write_nothing(curves_run, tmp_path, monkeypatch):
    before = sorted(p.name for p in curves_run.rglob("*"))
    ok(call("erasure_curves", {"path": str(curves_run)}))
    ok(call("sample_erasure_regions", REGIONS_ARGS))
    assert sorted(p.name for p in curves_run.rglob("*")) == before
    monkeypatch.chdir(tmp_path)
    ok(call("sample_erasure_regions", REGIONS_ARGS))
    assert not list(tmp_path.iterdir())


def test_job_tools_lifecycle(tmp_path):
    out_dir = str(tmp_path / "job")

    async def lifecycle(client):
        started = ok(await client.call_tool(
            "start_ensemble_job",
            {"spec": SPEC, "trajectories": 30, "output_dir": out_dir, "seed": 9, "chunk_size": 10},
        ))  # fmt: skip
        status = ok(await client.call_tool(
            "job_status", {"job_id": started["job_id"], "wait_seconds": 60}
        ))  # fmt: skip
        listed = ok(await client.call_tool("list_jobs", {}))
        loaded = ok(await client.call_tool("load_results", {"path": out_dir}))
        canceled = ok(await client.call_tool("cancel_job", {"job_id": started["job_id"]}))
        missing = await client.call_tool("job_status", {"job_id": "nope"})
        return started, status, listed, loaded, canceled, missing

    started, status, listed, loaded, canceled, missing = session(lifecycle)
    assert status["status"] == "completed" and status["trajectories_done"] == 30
    assert started["job_id"] in {j["job_id"] for j in listed["jobs"]}
    assert loaded["num_trajectories"] == 30
    assert loaded["purification"] == api.run_ensemble(SPEC, 30, seed=9)["purification"]
    assert canceled["status"] == "completed"  # canceling a finished job is harmless
    assert "unknown job_id" in error_text(missing)


def test_resources_and_prompts():
    async def read(client):
        resources = (await client.list_resources()).resources
        quick = await client.read_resource("docs://sparsegf2/quickstart")
        caps = await client.read_resource("docs://sparsegf2/capabilities")
        prompts = (await client.list_prompts()).prompts
        prompt = await client.get_prompt(
            "study_purification_transition", {"num_system": "48", "rates": "0.1, 0.2"}
        )
        erasure = await client.get_prompt("study_erasure_protection", {})
        return resources, quick, caps, prompts, prompt, erasure

    resources, quick, caps, prompts, prompt, erasure = session(read)
    uris = {str(r.uri) for r in resources}
    assert {"docs://sparsegf2/quickstart", "docs://sparsegf2/capabilities"} <= uris
    assert quick.contents[0].text == api.quickstart()
    assert json.loads(caps.contents[0].text)["version"] == api.capabilities()["version"]
    assert {p.name for p in prompts} == {
        "study_purification_transition",
        "study_erasure_protection",
    }
    text = prompt.messages[0].content.text
    assert "48 system qubits" in text and "0.1, 0.2" in text
    assert "erasure" in erasure.messages[0].content.text


def test_input_errors_come_back_as_messages_that_name_the_field(tmp_path, monkeypatch):
    # names that are not strings, and a size no graph could be generated for
    spec = dict(SPEC, graph={"family": ["cycle"]})
    text = error_text(call("run_ensemble", {"spec": spec, "trajectories": 2, "seed": 1}))
    assert "graph.family must be a string" in text
    spec = dict(SPEC, probes=[{"kind": ["x"]}])
    text = error_text(call("estimate_ensemble", {"spec": spec, "trajectories": 2}))
    assert "probes[0].kind must be a string" in text
    spec = dict(SPEC, num_system=10**12)
    text = error_text(call("estimate_ensemble", {"spec": spec, "trajectories": 2}))
    assert "num_system is too large" in text
    # an output_dir that cannot be created: below a regular file, or without permission
    (tmp_path / "a_file").write_text("", encoding="utf-8")
    real_mkdir = Path.mkdir

    def mkdir(self, *args, **kwargs):
        if self.name == "denied":
            raise PermissionError(13, "Permission denied", str(self))
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    for out, expected in (
        (tmp_path / "a_file" / "out", "is a file, not a directory"),
        (tmp_path / "denied", "is not writable: [Errno 13] Permission denied"),
    ):
        for tool, arguments in (
            ("run_ensemble", {"spec": SPEC, "trajectories": 2, "seed": 1}),
            ("start_ensemble_job", {"spec": SPEC, "trajectories": 2, "seed": 1}),
            ("run_circuit", {"circuit": "H 0\nM 0"}),
        ):
            text = error_text(call(tool, {**arguments, "output_dir": str(out)}))
            assert expected in text and "internal error" not in text, (tool, text)


def test_unanticipated_failures_carry_their_type_and_message(monkeypatch, caplog):
    def fail(error):
        def raising(*args, **kwargs):
            raise error

        monkeypatch.setattr(api, "run_ensemble", raising)
        caplog.clear()
        return error_text(call("run_ensemble", {"spec": SPEC, "trajectories": 2}))

    # the API raises ValueError for every input error, so anything else is a bug: a
    # TypeError too (a None that reaches a subscript, say), never the caller's to fix
    for error, shown in (
        (RuntimeError("kaboom"), "RuntimeError: kaboom"),
        (PermissionError(13, "Permission denied", "x"), "PermissionError: [Errno 13] "
                                                        "Permission denied: 'x'"),
        (TypeError("'NoneType' object is not subscriptable"),
         "TypeError: 'NoneType' object is not subscriptable"),
    ):  # fmt: skip
        assert fail(error).endswith(f"{shown} (internal error: please report it)")
        # and the server's log (stderr) keeps the traceback
        (logged,) = [r for r in caplog.records if r.name == "sparsegf2.mcp_server"]
        assert logged.levelname == "ERROR" and logged.exc_info[1] is error


@pytest.mark.parametrize(
    "tool, given, expected",
    [
        ("run_ensemble", {"trajectories": "3"}, "valid integer"),
        ("run_ensemble", {"trajectories": True}, "valid integer"),
        ("run_ensemble", {"trajectories": 2.5}, "valid integer"),
        ("run_ensemble", {"seed": True}, "valid integer"),
        ("run_ensemble", {"seed": "1"}, "valid integer"),
        ("run_ensemble", {"max_seconds": "inf"}, "valid number"),
        ("run_ensemble", {"max_seconds": True}, "valid number"),
        ("estimate_ensemble", {"trajectories": True}, "valid integer"),
        ("start_ensemble_job", {"chunk_size": True}, "valid integer"),
        ("start_ensemble_job", {"seed": "3"}, "valid integer"),
        ("run_circuit", {"shots": "2"}, "valid integer"),
        ("run_circuit", {"shots": True}, "valid integer"),
        ("run_circuit", {"phases": "false"}, "valid boolean"),
        ("run_circuit", {"phases": 0}, "valid boolean"),
        ("run_circuit", {"observe_ticks": "true"}, "valid boolean"),
        ("run_circuit", {"tick_stride": True}, "valid integer"),
        ("run_circuit", {"num_qubits": "4"}, "valid integer"),
        ("job_status", {"wait_seconds": True}, "valid number"),
        ("run_ensemble", {"num_threads": True}, "valid integer"),
        ("run_ensemble", {"num_threads": "2"}, "valid integer"),
        ("run_ensemble", {"num_threads": 2.5}, "valid integer"),
        ("start_ensemble_job", {"num_threads": False}, "valid integer"),
        ("start_ensemble_job", {"num_threads": "4"}, "valid integer"),
    ],
)
def test_integer_and_boolean_arguments_are_strict(tool, given, expected, tmp_path):
    # true used to run 1 trajectory, and "3" 3 (pydantic's lax mode); the API refuses both
    base = {
        "run_ensemble": {"spec": SPEC, "trajectories": 2, "seed": 1},
        "estimate_ensemble": {"spec": SPEC, "trajectories": 2},
        "start_ensemble_job": {"spec": SPEC, "trajectories": 2, "output_dir": str(tmp_path)},
        "run_circuit": {"circuit": "H 0\nTICK\nM 0", "seed": 1,
                        "observables": [{"kind": "entropy", "region": [0]}]},
        "job_status": {"job_id": "x"},
    }[tool]  # fmt: skip
    text = error_text(call(tool, {**base, **given}))
    (field,) = given
    assert f"\n{field}" in text and expected in text, text
    assert not (tmp_path / "chunks").exists() and not (tmp_path / "result.npz").exists()


def test_whole_number_floats_count_as_integers():
    # as in the API (and JSON Schema's "integer"): 4.0 is 4, but 4.5 is refused above
    out = ok(call("run_ensemble", {"spec": SPEC, "trajectories": 4.0, "seed": 4.0}))
    assert (out["num_trajectories"], out["seed"]) == (4, 4)
    assert ok(call("run_circuit", {"circuit": "H 0\nM 0", "shots": 2.0, "seed": 1}))["shots"] == 2
    schema = session(lambda c: c.list_tools()).tools
    ensemble = next(t for t in schema if t.name == "run_ensemble").input_schema["properties"]
    assert {"type": "integer"} in ensemble["trajectories"]["anyOf"]


def _num_threads_schema(tools, name: str) -> dict:
    return next(t for t in tools if t.name == name).input_schema["properties"]["num_threads"]


def test_num_threads_defaults_to_the_threads_this_process_may_use(monkeypatch):
    import numba

    tools = session(lambda c: c.list_tools()).tools
    for name in ("run_ensemble", "start_ensemble_job"):
        field = _num_threads_schema(tools, name)
        assert field["type"] == "integer" and field["default"] == numba.config.NUMBA_NUM_THREADS
        description = next(t for t in tools if t.name == name).description
        assert "num_threads" in description and str(numba.config.NUMBA_NUM_THREADS) in description
    # read when the server is built (Clank's sandbox sets NUMBA_NUM_THREADS=16 for its 16 CPUs)
    monkeypatch.setattr(numba.config, "NUMBA_NUM_THREADS", 3)
    tools = session(lambda c: c.list_tools()).tools
    assert [
        _num_threads_schema(tools, name)["default"]
        for name in ("run_ensemble", "start_ensemble_job")
    ] == [3, 3]


def test_num_threads_is_passed_to_the_api(monkeypatch, tmp_path):
    import numba

    seen = []

    def recorder(name):
        def record(*args, **kwargs):
            seen.append((name, kwargs["num_threads"]))
            return {"recorded": name}

        return record

    monkeypatch.setattr(api, "run_ensemble", recorder("run_ensemble"))
    monkeypatch.setattr(api, "start_ensemble_job", recorder("start_ensemble_job"))
    calls = {
        "run_ensemble": {"spec": SPEC, "trajectories": 2},
        "start_ensemble_job": {"spec": SPEC, "trajectories": 2, "output_dir": str(tmp_path)},
    }
    for tool, arguments in calls.items():
        assert ok(call(tool, arguments)) == {"recorded": tool}
        assert ok(call(tool, {**arguments, "num_threads": 2})) == {"recorded": tool}
    every = numba.config.NUMBA_NUM_THREADS
    assert seen == [("run_ensemble", every), ("run_ensemble", 2),
                    ("start_ensemble_job", every), ("start_ensemble_job", 2)]  # fmt: skip


@pytest.mark.parametrize("tool", ["run_ensemble", "start_ensemble_job"])
@pytest.mark.parametrize("value", [0, -3])
def test_num_threads_below_one_is_refused(tool, value, tmp_path):
    arguments = {"spec": SPEC, "trajectories": 2, "seed": 1, "num_threads": value}
    if tool == "start_ensemble_job":
        arguments["output_dir"] = str(tmp_path / "job")
    text = error_text(call(tool, arguments))
    assert f"num_threads must be at least 1, got {value}" in text, text
    assert not (tmp_path / "job").exists()


def test_num_threads_above_the_maximum_runs_on_the_maximum():
    import numba

    many = ok(call("run_ensemble", {"spec": SPEC, "trajectories": 6, "seed": 2,
                                    "num_threads": numba.config.NUMBA_NUM_THREADS + 7}))  # fmt: skip
    one = ok(call("run_ensemble", {"spec": SPEC, "trajectories": 6, "seed": 2, "num_threads": 1}))
    assert many["num_trajectories"] == 6
    # every result is the same at any thread count
    assert (many["probes"], many["purification"]) == (one["probes"], one["purification"])


def test_fresh_seeds_fit_in_a_double():
    seeds = [ok(call("run_circuit", {"circuit": "H 0\nM 0"}))["seed"] for _ in range(5)]
    assert all(0 <= seed < 2**53 for seed in seeds)


def test_main_help(capsys):
    assert mcp_server.main(["--help"]) == 0
    assert "claude mcp add" in capsys.readouterr().err


def test_stdio_transport_end_to_end():
    """The real server process over stdio: nothing but protocol messages on stdout."""
    import os
    import sys

    src = str(Path(mcp_server.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    params = mcp.StdioServerParameters(
        command=sys.executable, args=["-m", "sparsegf2.mcp_server"], env=env
    )

    big = 2**64 + 3  # an explicit seed of any size comes back exactly, through real JSON

    async def runner():
        async with mcp.Client(params) as client:
            first = await client.call_tool("run_circuit", {"circuit": "H 0\nM 0", "seed": 1})
            return first, await client.call_tool("run_circuit", {"circuit": "H 0", "seed": big})

    result, seeded = asyncio.run(asyncio.wait_for(runner(), timeout=120))
    assert ok(result)["measurements"] == api.run_circuit("H 0\nM 0", seed=1)["measurements"]
    assert ok(seeded)["seed"] == big and f'"seed":{big},' in seeded.content[0].text
