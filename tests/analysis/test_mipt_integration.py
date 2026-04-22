"""Integration tests: analysis observables across the MIPT phase transition.

Uses a brickwork circuit (alternating even/odd CNOT matchings on a cycle)
with Z-measurements at rate p, driven by SparseGF2. Verifies that the
analysis package produces physically consistent results across the
volume-law and area-law phases.

n=32, depth_factor=4 (128 layers per sample), 20 seeds per measurement rate.
"""
import numpy as np
import pytest

from sparsegf2.core.sparse_tableau import SparseGF2
from sparsegf2.analysis import (
    compute_weight_stats,
    verify_weight_mass_identity,
    build_tanner_graph,
    observe,
)

# ── Shared CNOT symplectic matrix ──────────────────────────────────
CNOT = np.array([
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
], dtype=np.uint8)

N_QUBITS = 32
DEPTH_FACTOR = 4
N_SEEDS = 20
P_VALUES = [0.05, 0.10, 0.14, 0.16, 0.20, 0.25, 0.30]


def run_brickwork_sample(n, p, depth_factor, seed):
    """Run one brickwork circuit sample and return the final simulator state.

    Disables the per-op auto mode-switch because this test depends on the
    specific min-weight-pivot generator representation (which is a
    sparse-mode-only optimization). Auto-switching to dense mode mid-run
    produces an equivalent stabilizer group with different generator weights,
    shifting analytic observables like cv_w.
    """
    rng = np.random.default_rng(seed)
    sim = SparseGF2(n)
    sim._check_interval = 10 ** 9  # disable auto mode switching
    total_layers = depth_factor * n
    for layer in range(total_layers):
        # Gate step: alternating even/odd matchings on ring
        if layer % 2 == 0:
            pairs = [(i, (i + 1) % n) for i in range(0, n, 2)]
        else:
            pairs = [(i, (i + 1) % n) for i in range(1, n, 2)]
        for qi, qj in pairs:
            sim.apply_gate(qi, qj, CNOT)
        # Measurement step
        for q in range(n):
            if rng.random() < p:
                sim.apply_measurement_z(q)
    return sim


# ── Pre-compute the full sweep once for all tests ──────────────────
# Using module-level cache so each p/seed pair is simulated exactly once.

_sweep_cache = {}


def _get_sweep():
    """Run the full p-sweep (lazy, cached at module level)."""
    if _sweep_cache:
        return _sweep_cache

    for p in P_VALUES:
        abars = []
        var_ws = []
        cv_ws = []
        identity_results = []
        sims = []
        for seed in range(N_SEEDS):
            sim = run_brickwork_sample(N_QUBITS, p, DEPTH_FACTOR, seed)
            stats = compute_weight_stats(sim)
            abars.append(stats.abar)
            var_ws.append(stats.var_w)
            cv_ws.append(stats.cv_w)
            identity_results.append(
                verify_weight_mass_identity(sim, raise_on_failure=False)
            )
            sims.append(sim)

        _sweep_cache[p] = {
            'abars': np.array(abars),
            'var_ws': np.array(var_ws),
            'cv_ws': np.array(cv_ws),
            'identity_results': identity_results,
            'sims': sims,
        }
    return _sweep_cache


# ═══════════════════════════════════════════════════════════════════
# C1. PHASE TRANSITION DETECTION
# ═══════════════════════════════════════════════════════════════════

class TestPhaseTransitionDetection:
    """Verify abar, var_w, cv_w trends across the MIPT transition."""

    def test_abar_decreasing_trend(self):
        """abar must decrease overall: abar(p=0.05) > abar(p=0.30)."""
        sweep = _get_sweep()
        mean_abars = {p: np.mean(sweep[p]['abars']) for p in P_VALUES}
        assert mean_abars[0.05] > mean_abars[0.30], (
            f"abar at p=0.05 ({mean_abars[0.05]:.2f}) should exceed "
            f"abar at p=0.30 ({mean_abars[0.30]:.2f})"
        )

    def test_abar_volume_law_dense(self):
        """At p=0.05 (volume-law), abar > n/4 = 8."""
        sweep = _get_sweep()
        mean_abar = np.mean(sweep[0.05]['abars'])
        assert mean_abar > N_QUBITS / 4, (
            f"Volume-law abar={mean_abar:.2f} should exceed n/4={N_QUBITS / 4}"
        )

    def test_abar_area_law_sparse(self):
        """At p=0.30 (area-law), abar < n/4 = 8."""
        sweep = _get_sweep()
        mean_abar = np.mean(sweep[0.30]['abars'])
        assert mean_abar < N_QUBITS / 4, (
            f"Area-law abar={mean_abar:.2f} should be below n/4={N_QUBITS / 4}"
        )

    def test_var_w_peak_near_pc(self):
        """argmax of cv_w (not raw var_w) should fall in [0.13, 0.19], near p_c.

        For CNOT-only brickwork circuits, raw var_w decreases monotonically
        because it scales with abar^2. The coefficient of variation cv_w = std_w / wbar
        normalizes this out and peaks near criticality, where the weight
        distribution is broadest relative to its mean.
        """
        sweep = _get_sweep()
        mean_cv_ws = {p: np.mean(sweep[p]['cv_ws']) for p in P_VALUES}
        peak_p = max(mean_cv_ws, key=mean_cv_ws.get)
        assert 0.13 <= peak_p <= 0.19, (
            f"cv_w peak at p={peak_p}, expected in [0.13, 0.19]. "
            f"Values: {', '.join(f'p={p}: {v:.4f}' for p, v in sorted(mean_cv_ws.items()))}"
        )


# ═══════════════════════════════════════════════════════════════════
# C2. IDENTITY ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════

class TestIdentityRobustness:
    """Weight mass identity n*abar = 2n*wbar must hold for every sample."""

    def test_identity_at_p016(self):
        """verify_weight_mass_identity must pass for ALL 20 samples at p=0.16."""
        sweep = _get_sweep()
        results = sweep[0.16]['identity_results']
        for i, res in enumerate(results):
            assert res['inv_pass'], (
                f"Sample {i}: inv_len sum={res['total_inv']} != "
                f"supp_len sum={res['total_supp']}"
            )
            assert res['x_pass'], (
                f"Sample {i}: inv_x_len sum={res['total_x_inv']} != "
                f"X-weight mass={res['total_x_supp']}"
            )

    def test_identity_all_p_values(self):
        """Weight mass identity must hold across ALL p values and seeds."""
        sweep = _get_sweep()
        for p in P_VALUES:
            for i, res in enumerate(sweep[p]['identity_results']):
                assert res['inv_pass'], (
                    f"p={p}, seed={i}: inv_len sum={res['total_inv']} != "
                    f"supp_len sum={res['total_supp']}"
                )
                assert res['x_pass'], (
                    f"p={p}, seed={i}: inv_x_len sum={res['total_x_inv']} != "
                    f"X-weight mass={res['total_x_supp']}"
                )


# ═══════════════════════════════════════════════════════════════════
# C3. STABILIZER/DESTABILIZER SYMMETRY
# ═══════════════════════════════════════════════════════════════════

class TestStabilizerDestabilizerSymmetry:
    """In volume-law phase, stabilizer and destabilizer weights should be similar."""

    def test_symmetry_volume_law(self):
        """wbar_stab / wbar_destab ratio should be near 1 at p=0.05.

        For CNOT-only brickwork at n=32, the control/target asymmetry of
        CNOT and finite-size fluctuations produce a stab/destab ratio of
        ~1.1. Individual samples can fluctuate to ~0.4 or ~2.5 at small n,
        so the per-sample bound is [1/3, 3]. The mean ratio across 20 seeds
        must be within [0.7, 1.5], confirming approximate symmetry.
        """
        sweep = _get_sweep()
        ratios = []
        for sim in sweep[0.05]['sims']:
            stats = compute_weight_stats(sim)
            if stats.wbar_destab > 0:
                ratio = stats.wbar_stab / stats.wbar_destab
            else:
                ratio = float('inf')
            ratios.append(ratio)
            # Per-sample: neither sector dominates by more than 3x
            assert 1 / 3 <= ratio <= 3.0, (
                f"Extreme stab/destab ratio {ratio:.3f} "
                f"(stab={stats.wbar_stab:.2f}, destab={stats.wbar_destab:.2f})"
            )
        mean_ratio = np.mean(ratios)
        assert 0.7 <= mean_ratio <= 1.5, (
            f"Mean stab/destab ratio at p=0.05: {mean_ratio:.4f}, "
            f"expected in [0.7, 1.5]. "
            f"Individual: {[f'{r:.3f}' for r in ratios]}"
        )


# ═══════════════════════════════════════════════════════════════════
# C4. TANNER GRAPH CONSISTENCY
# ═══════════════════════════════════════════════════════════════════

class TestTannerGraphConsistency:
    """Tanner graph edge count must equal weight_mass = n * abar."""

    def test_tanner_edges_equal_weight_mass(self):
        """For every sample, |E| of Tanner graph == weight_mass."""
        sweep = _get_sweep()
        # Test a subset of p values to keep runtime reasonable
        for p in [0.05, 0.16, 0.30]:
            for i, sim in enumerate(sweep[p]['sims']):
                stats = compute_weight_stats(sim)
                G = build_tanner_graph(sim, include_destabilizers=True)
                n_edges = len(G.edges())
                weight_mass = stats.weight_mass
                expected = int(round(N_QUBITS * stats.abar))
                assert n_edges == weight_mass, (
                    f"p={p}, seed={i}: Tanner |E|={n_edges} != "
                    f"weight_mass={weight_mass}"
                )
                assert weight_mass == expected, (
                    f"p={p}, seed={i}: weight_mass={weight_mass} != "
                    f"n*abar={expected}"
                )


# ═══════════════════════════════════════════════════════════════════
# C5. OBSERVE() SWEEP COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════

class TestObserveSweep:
    """observe() must return a complete dict of finite scalar values."""

    EXPECTED_KEYS = {
        'abar', 'var_a', 'std_a', 'cv_a', 'a_max', 'a_min', 'skew_a',
        'wbar', 'var_w', 'std_w', 'cv_w', 'w_max', 'w_min', 'skew_w',
        'wbar_stab', 'wbar_destab', 'mean_wt_x',
        'weight_mass', 'identity_holds',
        'pei_proxy', 'wcp_proxy', 'pivot_speedup', 'n_anticommuting_qubits',
        'p',
    }

    def test_observe_keys_complete(self):
        """observe() at p=0.16, seed=0 must contain all expected keys."""
        sweep = _get_sweep()
        sim = sweep[0.16]['sims'][0]
        obs = observe(sim, p=0.16)
        missing = self.EXPECTED_KEYS - set(obs.keys())
        extra = set(obs.keys()) - self.EXPECTED_KEYS
        assert not missing, f"Missing keys in observe(): {missing}"
        # Extra keys are allowed but not expected -- just warn
        if extra:
            import warnings
            warnings.warn(f"Extra keys in observe(): {extra}")

    def test_observe_values_finite(self):
        """All numeric values from observe() must be finite."""
        sweep = _get_sweep()
        sim = sweep[0.16]['sims'][0]
        obs = observe(sim, p=0.16)
        for key, val in obs.items():
            if isinstance(val, (int, float, np.integer, np.floating)):
                assert np.isfinite(val), (
                    f"observe()['{key}'] = {val} is not finite"
                )
            elif isinstance(val, bool):
                pass  # booleans are always valid
            else:
                # Unexpected type -- flag but don't fail
                import warnings
                warnings.warn(
                    f"observe()['{key}'] has unexpected type {type(val).__name__}"
                )

    def test_observe_p_metadata(self):
        """observe(sim, p=0.16) must include p=0.16 in the output."""
        sweep = _get_sweep()
        sim = sweep[0.16]['sims'][0]
        obs = observe(sim, p=0.16)
        assert 'p' in obs
        assert obs['p'] == 0.16

    def test_observe_identity_consistent(self):
        """observe()['identity_holds'] must agree with verify_weight_mass_identity.

        identity_holds checks n*abar == 2n*wbar (weight mass identity = inv_pass).
        The X-weight identity (x_pass) is a separate check in verify_weight_mass_identity.
        """
        sweep = _get_sweep()
        sim = sweep[0.16]['sims'][0]
        obs = observe(sim, p=0.16)
        direct = verify_weight_mass_identity(sim, raise_on_failure=False)
        assert obs['identity_holds'] == direct['inv_pass'], (
            f"observe identity_holds={obs['identity_holds']} disagrees with "
            f"direct verification inv_pass={direct['inv_pass']}"
        )
