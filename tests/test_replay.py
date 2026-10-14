"""Replay: probes and checkpoints never change a trajectory (the randomness contract)."""

import numpy as np
import pytest

from sparsegf2 import graphs, mipt


def _spec(**kw):
    n = 16
    base = dict(
        num_system=n,
        edges=[graphs.watts_strogatz(n, 4, 0.2, seed=(0, g)) for g in range(3)],
        circuits_per_graph=2,
        layers=24,
        measurement_rate=0.15,
        schedule="random_edges",
        reference="purification",
        checkpoints="final",
        probes=[mipt.code_snapshot()],
    )
    base.update(kw)
    return mipt.MonitoredCircuit(**base)


def test_contract_constant():
    assert mipt.RANDOMNESS_CONTRACT == 1


@pytest.mark.parametrize("t", [0, 3, 5])
def test_replay_with_new_probes_reproduces_the_code_bit_for_bit(t):
    spec = _spec()
    full = spec.run(seed=9)
    again = spec.replay(
        t,
        seed=9,
        checkpoints=8,
        probes=[
            mipt.code_dimension(),
            mipt.erasure_first_failure(20, checkpoints=[8, 16]),
            mipt.code_snapshot(checkpoints="final"),
        ],
    )
    assert again.num_trajectories == 1 and int(again.trajectories[0]) == t
    np.testing.assert_array_equal(
        full.code(t, 0).stabilizer_matrix(), again.code(0, 0).stabilizer_matrix()
    )
    assert again.values["code_dimension"][0, -1] == full.code(t, 0).num_logical


def test_replay_validates_the_index():
    with pytest.raises(ValueError):
        _spec().replay(6, seed=1)  # 3 graphs x 2 circuits = trajectories 0..5
