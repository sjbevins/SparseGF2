"""Simulator.run and sample(): exact replay against Stim, and noise statistics."""

import numpy as np
import pytest
import stim

from sparsegf2_v3 import Circuit, PhaseFreeError, Simulator, all_gates, sample

GATES_1Q = [g.name for g in all_gates("unitary") if g.num_qubits == 1]
GATES_2Q = [g.name for g in all_gates("unitary") if g.num_qubits == 2]


def random_noiseless_circuit(n: int, num_ops: int, rng: np.random.Generator) -> Circuit:
    """A random circuit using every noiseless instruction the run loop supports.

    Plain resets appear only at the start, where every qubit is unentangled, so
    their internal collapse cannot introduce randomness absent from the record.
    """
    lines = ["R " + " ".join(map(str, range(n)))]
    if n > 1:
        lines.append("RX " + str(int(rng.integers(n))))
    num_meas = 0
    num_obs = 0

    def q() -> int:
        return int(rng.integers(n))

    def pair() -> str:
        a, b = rng.choice(n, 2, replace=False)
        return f"{a} {b}"

    def bang() -> str:
        return "!" if rng.random() < 0.3 else ""

    for _ in range(num_ops):
        r = rng.random()
        if r < 0.22:
            lines.append(f"{GATES_1Q[rng.integers(len(GATES_1Q))]} {q()} {q()}")
        elif r < 0.40 and n > 1:
            lines.append(f"{GATES_2Q[rng.integers(len(GATES_2Q))]} {pair()}")
        elif r < 0.48 and n > 1:
            lines.append(f"CLIFFORD_2Q({int(rng.integers(11520))}) {pair()}")
        elif r < 0.62:
            name = ["M", "MX", "MY", "MR", "MRX", "MRY"][rng.integers(6)]
            lines.append(f"{name} {bang()}{q()} {bang()}{q()}")
            num_meas += 2
        elif r < 0.68 and n > 1:
            name = ["MXX", "MYY", "MZZ"][rng.integers(3)]
            a, b = rng.choice(n, 2, replace=False)
            lines.append(f"{name} {bang()}{a} {b}")
            num_meas += 1
        elif r < 0.76:
            k = int(rng.integers(1, min(n, 4) + 1))
            factors = "*".join(
                f"{bang()}{'XYZ'[rng.integers(3)]}{v}" for v in rng.choice(n, k, replace=False)
            )
            lines.append(f"MPP {factors} {'XYZ'[rng.integers(3)]}{q()}")
            num_meas += 2
        elif r < 0.84 and num_meas > 0 and n > 1:
            k = int(rng.integers(1, min(num_meas, 5) + 1))
            gate = ["CX", "CY", "CZ"][rng.integers(3)]
            lines.append(f"{gate} rec[-{k}] {q()}")
        elif r < 0.90 and num_meas > 1:
            k1, k2 = rng.choice(np.arange(1, min(num_meas, 6) + 1), 2, replace=False)
            lines.append(f"DETECTOR rec[-{k1}] rec[-{k2}]")
        elif r < 0.94 and num_meas > 0:
            lines.append(f"OBSERVABLE_INCLUDE({num_obs % 3}) rec[-1]")
            num_obs += 1
        elif r < 0.97:
            # a repeated measurement: deterministic detector and observable
            basis = ["M", "MX", "MY"][rng.integers(3)]
            qq = q()
            lines.append(f"{basis} {qq}\n{basis} {bang()}{qq}\nDETECTOR rec[-1] rec[-2]")
            lines.append(f"OBSERVABLE_INCLUDE({num_obs % 3}) rec[-1] rec[-2]")
            num_obs += 1
            num_meas += 2
        else:
            lines.append("TICK")
    return Circuit("\n".join(lines))


def stim_replay(circuit: Circuit, n: int, seed: int):
    ref = stim.TableauSimulator(seed=seed)
    ref.set_num_qubits(n)
    stim_circuit = circuit.to_stim()
    ref.do_circuit(stim_circuit)
    record = np.array(ref.current_measurement_record(), dtype=np.uint8)
    return ref, stim_circuit, record


def _deterministic_masks(stim_circuit: stim.Circuit, n: int, runs: int = 12):
    """Which detectors / observables have a fixed parity across noiseless Stim runs."""
    dets, obs = [], []
    for seed in range(runs):
        ref = stim.TableauSimulator(seed=1000 + seed)
        ref.set_num_qubits(n)
        ref.do_circuit(stim_circuit)
        rec = np.array(ref.current_measurement_record(), dtype=bool)[None, :]
        d, o = stim_circuit.compile_m2d_converter(skip_reference_sample=True).convert(
            measurements=rec, separate_observables=True
        )
        dets.append(d[0])
        obs.append(o[0])
    dets, obs = np.array(dets), np.array(obs)
    return np.all(dets == dets[0], axis=0), np.all(obs == obs[0], axis=0)


@pytest.mark.parametrize("n", [1, 2, 3, 6, 17, 64, 65, 100])
def test_replaying_stim_records_reproduces_everything(n, rng):
    compared = 0
    for trial in range(6 if n <= 17 else 2):
        circuit = random_noiseless_circuit(n, 120, rng)
        ref, stim_circuit, record = stim_replay(circuit, n, seed=trial)
        sim = Simulator(n, seed=trial + 100)
        result = sim.run(circuit, coins=record)
        np.testing.assert_array_equal(result.measurements, record)
        assert [str(p) for p in sim.stabilizers()] == [str(p) for p in ref.canonical_stabilizers()]
        if stim_circuit.num_detectors or stim_circuit.num_observables:
            # Detection events are only meaningful for detectors whose parity is
            # fixed in noiseless runs; for those, any reference sample gives the
            # same events, so ours must equal Stim's exactly.
            det_ok, obs_ok = _deterministic_masks(stim_circuit, n)
            dets, obs = stim_circuit.compile_m2d_converter().convert(
                measurements=record[None, :].astype(bool), separate_observables=True
            )
            np.testing.assert_array_equal(result.detectors[det_ok], dets[0][det_ok])
            np.testing.assert_array_equal(result.observables[obs_ok], obs[0][obs_ok])
            compared += int(det_ok.sum())
    assert compared > 0, "no deterministic detectors were compared"


def test_repeat_blocks_execute_like_stim(rng):
    circuit = Circuit(
        """
        H 0
        REPEAT 5 {
            CX 0 1
            M 1
            CX rec[-1] 2
            DETECTOR rec[-1]
            REPEAT 2 {
                H 0 2
                MX !2
            }
        }
        """
    )
    ref, stim_circuit, record = stim_replay(circuit, 3, seed=4)
    sim = Simulator(3)
    result = sim.run(circuit, coins=record)
    np.testing.assert_array_equal(result.measurements, record)
    assert [str(p) for p in sim.stabilizers()] == [str(p) for p in ref.canonical_stabilizers()]


def test_random_flags_match_interactive_measurement(rng):
    n = 12
    for _ in range(5):
        circuit = Circuit()
        expected = []
        mirror = Simulator(n, seed=0)
        for _ in range(80):
            if rng.random() < 0.6:
                a, b = (int(v) for v in rng.choice(n, 2, replace=False))
                index = int(rng.integers(11520))
                circuit.append("CLIFFORD_2Q", [a, b], [index])
                mirror.clifford_2q(index, a, b)
            else:
                qq = int(rng.integers(n))
                circuit.append("M", [qq])
                expected.append(not mirror.is_deterministic(f"Z{qq}"))
                mirror.measure(qq)
        for phases in (True, False):
            result = Simulator(n, phases=phases, seed=1).run(circuit)
            np.testing.assert_array_equal(result.random, expected)


def test_phase_free_run_hides_outcomes_but_reports_randomness(rng):
    circuit = random_noiseless_circuit(8, 100, rng)
    phased = Simulator(8, seed=3).run(circuit)
    free = Simulator(8, phases=False, seed=3).run(circuit)
    np.testing.assert_array_equal(free.random, phased.random)
    for attr in ("measurements", "detectors", "observables"):
        with pytest.raises(PhaseFreeError):
            getattr(free, attr)


def test_run_is_reproducible_from_seed(rng):
    circuit = random_noiseless_circuit(10, 150, rng)
    a = Simulator(10, seed=42).run(circuit).measurements
    b = Simulator(10, seed=42).run(circuit).measurements
    np.testing.assert_array_equal(a, b)


def test_coins_validation():
    sim = Simulator(2)
    circuit = Circuit("H 0\nM 0 1")
    with pytest.raises(ValueError, match="one entry per measurement"):
        sim.run(circuit, coins=[0])
    with pytest.raises(ValueError, match="0 or 1"):
        sim.run(circuit, coins=[0, 2])
    with pytest.raises(ValueError, match="qubits"):
        Simulator(1).run(circuit)


def test_observe_callback_at_ticks():
    circuit = Circuit(
        """
        H 0
        TICK
        CX 0 1
        TICK
        CX 1 2
        TICK
        M 0
        TICK
        """
    )
    sim = Simulator(3, seed=0)
    result = sim.run(circuit, observe=lambda s, tick: s.entropy([0]))
    assert result.observations == [(0, 0), (1, 1), (2, 1), (3, 0)]


def test_observe_resumes_with_correct_record_offsets():
    circuit = Circuit("H 0\nM 0\nTICK\nCX rec[-1] 1\nM 1\nDETECTOR rec[-1] rec[-2]")
    for seed in range(10):
        result = Simulator(2, seed=seed).run(circuit, observe=lambda s, t: None)
        assert result.measurements[0] == result.measurements[1]
        assert result.detectors[0] == 0


# -- noise: statistics against Stim's sampler -----------------------------------------------


def _compare_marginals(circuit_text: str, shots: int = 40_000, seed: int = 0):
    circuit = Circuit(circuit_text)
    ours = sample(circuit, shots, seed=seed).measurements.astype(float)
    theirs = stim.Circuit(circuit_text).compile_sampler(seed=seed).sample(shots).astype(float)
    for col in range(ours.shape[1]):
        p = theirs[:, col].mean()
        sigma = max(np.sqrt(p * (1 - p) / shots), 1e-4) * np.sqrt(2)
        assert abs(ours[:, col].mean() - p) < 6 * sigma, f"column {col}"
    # pairwise correlations (catches wrong 2-qubit Pauli orderings)
    if ours.shape[1] >= 2:
        for i in range(ours.shape[1]):
            for j in range(i + 1, ours.shape[1]):
                a = (ours[:, i] * ours[:, j]).mean()
                b = (theirs[:, i] * theirs[:, j]).mean()
                assert abs(a - b) < 6 * np.sqrt(max(b, 1e-3) / shots) * np.sqrt(2)


@pytest.mark.parametrize(
    "text",
    [
        "X_ERROR(0.1) 0\nM 0",
        "H 0\nY_ERROR(0.2) 0\nH 0\nM 0",
        "H 0\nZ_ERROR(0.3) 0\nH 0\nM 0",
        "DEPOLARIZE1(0.3) 0\nM 0\nH 1\nDEPOLARIZE1(0.3) 1\nH 1\nM 1",
        "PAULI_CHANNEL_1(0.1, 0.2, 0.3) 0\nM 0\nH 1\nPAULI_CHANNEL_1(0.1, 0.2, 0.3) 1\nH 1\nM 1",
        "DEPOLARIZE2(0.4) 0 1\nM 0 1",
        "H 0 1\nDEPOLARIZE2(0.4) 0 1\nH 0 1\nM 0 1",
        "PAULI_CHANNEL_2(0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.0,0.05) 0 1\nM 0 1\n",
        "H 0 1\nPAULI_CHANNEL_2(0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.0,0.05) 0 1\nH 0 1\nM 0 1",
        "M(0.2) 0\nMX(0.1) 1\nMR(0.05) 2\nMPP(0.15) Z0*Z1",
        "H 0\nCX 0 1\nX_ERROR(0.25) 0\nM 0 1\nCX rec[-1] 2\nM 2",
    ],
)
def test_noise_statistics_match_stim(text):
    _compare_marginals(text)


def test_noiseless_surface_code_has_no_detection_events():
    circuit = Circuit.from_stim(
        stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=4)
    )
    result = sample(circuit, 500, seed=1)
    assert not result.detectors.any()
    assert not result.observables.any()


def test_noisy_surface_code_detection_rates_match_stim():
    stim_circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
        before_round_data_depolarization=0.01,
    )
    shots = 30_000
    ours = sample(Circuit.from_stim(stim_circuit), shots, seed=7).detectors.astype(float)
    theirs = stim_circuit.compile_detector_sampler(seed=7).sample(shots).astype(float)
    p_ours, p_theirs = ours.mean(axis=0), theirs.mean(axis=0)
    sigma = np.sqrt(np.maximum(p_theirs, 1e-3) / shots) * np.sqrt(2)
    assert np.all(np.abs(p_ours - p_theirs) < 6 * sigma)


# -- sample() ------------------------------------------------------------------------------


def test_sample_is_reproducible_and_thread_independent(rng):
    circuit = random_noiseless_circuit(9, 120, rng)
    circuit.append_text("DEPOLARIZE1(0.1) 0 1 2\nM 0 1 2")
    a = sample(circuit, 300, seed=11, num_threads=1).measurements
    b = sample(circuit, 300, seed=11, num_threads=4).measurements
    c = sample(circuit, 300, seed=12).measurements
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sample_phase_free_and_zero_shots():
    circuit = Circuit("H 0\nM 0")
    free = sample(circuit, 10, phases=False)
    assert free.random.all()
    with pytest.raises(PhaseFreeError):
        _ = free.measurements
    assert sample(circuit, 0).measurements.shape == (0, 1)
