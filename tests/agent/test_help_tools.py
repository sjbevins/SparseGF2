"""help_topic and help_list."""

import pytest

from sparsegf2.agent.tools.help import TOPICS


@pytest.mark.parametrize("topic", TOPICS)
def test_every_topic_renders(run_tool, topic):
    out = run_tool("help_topic", topic=topic)
    assert not out.is_error and out.text.startswith(f"help_topic: {topic}")
    assert 200 < len(out.text) <= 4000


@pytest.mark.parametrize(
    "what, expect",
    [
        ("probes", "erasure_first_failure(samples)"),
        ("observables", "entropy(region)"),
        ("graph_families", "watts_strogatz(graph_k, graph_beta)"),
        ("gates", "CX"),
        ("instructions", "MPP"),
        ("edge_weight_families", "power_law(alpha, periodic)"),
        ("protocol_parameters", "measurement_rate"),
        ("tools", "help_topic"),
        ("stacks", "core"),
    ],
)
def test_lists(run_tool, what, expect):
    out = run_tool("help_list", what=what)
    assert not out.is_error and out.text.startswith(f"help_list: {what}") and expect in out.text


def test_bad_topic_is_a_tool_error(run_tool):
    out = run_tool("help_topic", topic="nonsense")
    assert out.is_error and "must be one of" in out.text
