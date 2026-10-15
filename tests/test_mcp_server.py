"""The MCP server, driven by an in-process MCP client (skipped if the mcp extra is missing)."""

import asyncio
import json

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
    "start_ensemble_job",
    "job_status",
    "cancel_job",
    "list_jobs",
    "load_results",
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
    text = error_text(call("run_ensemble", {"spec": big, "trajectories": 1_000_000}))
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
        cancelled = ok(await client.call_tool("cancel_job", {"job_id": started["job_id"]}))
        missing = await client.call_tool("job_status", {"job_id": "nope"})
        return started, status, listed, loaded, cancelled, missing

    started, status, listed, loaded, cancelled, missing = session(lifecycle)
    assert status["status"] == "completed" and status["trajectories_done"] == 30
    assert started["job_id"] in {j["job_id"] for j in listed["jobs"]}
    assert loaded["num_trajectories"] == 30
    assert loaded["purification"] == api.run_ensemble(SPEC, 30, seed=9)["purification"]
    assert cancelled["status"] == "completed"  # cancelling a finished job is harmless
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


def test_input_errors_come_back_as_messages_that_name_the_field(tmp_path):
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
    # an output_dir that cannot be created (below a regular file)
    (tmp_path / "a_file").write_text("", encoding="utf-8")
    out = str(tmp_path / "a_file" / "out")
    for tool, arguments in (
        ("run_ensemble", {"spec": SPEC, "trajectories": 2, "seed": 1, "output_dir": out}),
        ("start_ensemble_job", {"spec": SPEC, "trajectories": 2, "seed": 1, "output_dir": out}),
        ("run_circuit", {"circuit": "H 0\nM 0", "output_dir": out}),
    ):
        assert "is not writable" in error_text(call(tool, arguments)), tool


def test_unanticipated_failures_carry_their_type_and_message(monkeypatch):
    def fail(error):
        def raising(*args, **kwargs):
            raise error

        monkeypatch.setattr(api, "run_ensemble", raising)
        return error_text(call("run_ensemble", {"spec": SPEC, "trajectories": 2}))

    text = fail(RuntimeError("kaboom"))
    assert text.endswith("RuntimeError: kaboom (internal error: please report it)")
    text = fail(PermissionError(13, "Permission denied", "x"))
    assert text.endswith("PermissionError: [Errno 13] Permission denied: 'x' (internal error: "
                         "please report it)")  # fmt: skip
    # a TypeError is an input error (wrong argument types), not a bug to report
    assert fail(TypeError("regions must be lists")).endswith("TypeError: regions must be lists")


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
    from pathlib import Path

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
