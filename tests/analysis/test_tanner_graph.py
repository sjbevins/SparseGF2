"""Tests for sparsegf2.analysis.tanner_graph module.

Tests B1-B7: bipartite structure, edge count consistency, Pauli annotations,
hypergraph representation, degree annotations, visualization smoke, node types.
"""
import tempfile
import os

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from sparsegf2.core.sparse_tableau import SparseGF2
from sparsegf2.analysis.tanner_graph import (
    build_tanner_graph,
    build_tanner_hypergraph,
    plot_tanner_graph,
)
from sparsegf2.analysis.weight_stats import compute_weight_stats

# Gate symplectic matrices
CNOT = np.array([[1,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,1]], dtype=np.uint8)
CZ   = np.array([[1,0,0,1],[0,1,1,0],[0,0,1,0],[0,0,0,1]], dtype=np.uint8)


def _make_sim_n4_initial():
    """n=4 simulator at initial state (no gates applied)."""
    return SparseGF2(4)


def _make_sim_n4_cnot01():
    """n=4 after CNOT(0,1)."""
    sim = SparseGF2(4)
    sim.apply_gate(0, 1, CNOT)
    return sim


def _make_sim_n4_cnot01_cz23():
    """n=4 after CNOT(0,1) then CZ(2,3)."""
    sim = SparseGF2(4)
    sim.apply_gate(0, 1, CNOT)
    sim.apply_gate(2, 3, CZ)
    return sim


# ---------------------------------------------------------------------------
# B1. BIPARTITE STRUCTURE TEST
# ---------------------------------------------------------------------------
class TestB1BipartiteStructure:
    """Initial n=4: 4 variable + 8 check = 12 nodes, weight_mass=8 edges,
    all variable degree 2, all check degree 1, graph is bipartite."""

    def test_node_count(self):
        sim = _make_sim_n4_initial()
        G = build_tanner_graph(sim)
        n = 4
        # 4 variable nodes + 2*4=8 check nodes = 12
        assert len(G.nodes()) == n + 2 * n  # 12

    def test_edge_count_initial(self):
        sim = _make_sim_n4_initial()
        G = build_tanner_graph(sim)
        stats = compute_weight_stats(sim)
        # Initial: each of 2n=8 generators has weight 1, so weight_mass = 8
        assert stats.weight_mass == 8
        assert len(G.edges()) == 8

    def test_variable_nodes_degree_2(self):
        sim = _make_sim_n4_initial()
        G = build_tanner_graph(sim)
        for q in range(4):
            assert G.degree(('v', q)) == 2, (
                f"Variable node v{q} should have degree 2, got {G.degree(('v', q))}")

    def test_check_nodes_degree_1(self):
        sim = _make_sim_n4_initial()
        G = build_tanner_graph(sim)
        for r in range(8):
            assert G.degree(('c', r)) == 1, (
                f"Check node c{r} should have degree 1, got {G.degree(('c', r))}")

    def test_is_bipartite(self):
        import networkx as nx
        sim = _make_sim_n4_initial()
        G = build_tanner_graph(sim)
        assert nx.is_bipartite(G)


# ---------------------------------------------------------------------------
# B2. EDGE COUNT CONSISTENCY
# ---------------------------------------------------------------------------
class TestB2EdgeCountConsistency:
    """After each circuit step, len(G.edges()) == weight_mass."""

    def test_initial(self):
        sim = _make_sim_n4_initial()
        G = build_tanner_graph(sim)
        stats = compute_weight_stats(sim)
        assert len(G.edges()) == stats.weight_mass

    def test_after_cnot01(self):
        sim = _make_sim_n4_cnot01()
        G = build_tanner_graph(sim)
        stats = compute_weight_stats(sim)
        # CNOT(0,1): g0=X0X1(wt2), g5=Z0Z1(wt2), rest wt1 => mass=10
        assert stats.weight_mass == 10
        assert len(G.edges()) == stats.weight_mass

    def test_after_cnot01_cz23(self):
        sim = _make_sim_n4_cnot01_cz23()
        G = build_tanner_graph(sim)
        stats = compute_weight_stats(sim)
        # +CZ(2,3): g2=X2Z3(wt2), g3=Z2X3(wt2) => mass=12
        assert stats.weight_mass == 12
        assert len(G.edges()) == stats.weight_mass


# ---------------------------------------------------------------------------
# B3. PAULI ANNOTATION TEST
# ---------------------------------------------------------------------------
class TestB3PauliAnnotation:
    """Verify Pauli edge labels match PLT encoding (I=0, Z=1, X=2, Y=3)."""

    def test_initial_destab_X(self):
        """Destab g0 at initial: plt[0,0]=2 (X), edge (v0,c0) pauli='X'."""
        sim = _make_sim_n4_initial()
        G = build_tanner_graph(sim)
        edge_data = G.get_edge_data(('v', 0), ('c', 0))
        assert edge_data is not None, "Edge (v0, c0) should exist"
        assert edge_data['pauli'] == 'X'
        assert edge_data['pauli_val'] == 2

    def test_initial_stab_Z(self):
        """Stab g4 at initial: plt[4,0]=1 (Z), edge (v0,c4) pauli='Z'."""
        sim = _make_sim_n4_initial()
        G = build_tanner_graph(sim)
        edge_data = G.get_edge_data(('v', 0), ('c', 4))
        assert edge_data is not None, "Edge (v0, c4) should exist"
        assert edge_data['pauli'] == 'Z'
        assert edge_data['pauli_val'] == 1

    def test_cnot_spreads_X(self):
        """After CNOT(0,1): g0=X0X1, both edges from c0 should be pauli='X'."""
        sim = _make_sim_n4_cnot01()
        G = build_tanner_graph(sim)
        e0 = G.get_edge_data(('v', 0), ('c', 0))
        e1 = G.get_edge_data(('v', 1), ('c', 0))
        assert e0 is not None, "Edge (v0, c0) should exist after CNOT"
        assert e1 is not None, "Edge (v1, c0) should exist after CNOT"
        assert e0['pauli'] == 'X'
        assert e1['pauli'] == 'X'

    def test_cnot_spreads_Z(self):
        """After CNOT(0,1): g5=Z0Z1 (stab for q1), both edges pauli='Z'."""
        sim = _make_sim_n4_cnot01()
        n = 4
        G = build_tanner_graph(sim)
        # g_{n+1} = g5 starts as Z1, after CNOT(0,1) becomes Z0Z1
        e0 = G.get_edge_data(('v', 0), ('c', n + 1))
        e1 = G.get_edge_data(('v', 1), ('c', n + 1))
        assert e0 is not None, "Edge (v0, c5) should exist"
        assert e1 is not None, "Edge (v1, c5) should exist"
        assert e0['pauli'] == 'Z'
        assert e1['pauli'] == 'Z'

    def test_cz_mixed_paulis(self):
        """After CZ(2,3): g2=X2Z3, edge to v2 pauli='X', edge to v3 pauli='Z'."""
        sim = _make_sim_n4_cnot01_cz23()
        G = build_tanner_graph(sim)
        e2 = G.get_edge_data(('v', 2), ('c', 2))
        e3 = G.get_edge_data(('v', 3), ('c', 2))
        assert e2 is not None, "Edge (v2, c2) should exist"
        assert e3 is not None, "Edge (v3, c2) should exist"
        assert e2['pauli'] == 'X', f"Expected X, got {e2['pauli']}"
        assert e3['pauli'] == 'Z', f"Expected Z, got {e3['pauli']}"

    def test_cz_reverse_mixed(self):
        """After CZ(2,3): g3=Z2X3, edge to v2 pauli='Z', edge to v3 pauli='X'."""
        sim = _make_sim_n4_cnot01_cz23()
        G = build_tanner_graph(sim)
        e2 = G.get_edge_data(('v', 2), ('c', 3))
        e3 = G.get_edge_data(('v', 3), ('c', 3))
        assert e2 is not None, "Edge (v2, c3) should exist"
        assert e3 is not None, "Edge (v3, c3) should exist"
        assert e2['pauli'] == 'Z', f"Expected Z, got {e2['pauli']}"
        assert e3['pauli'] == 'X', f"Expected X, got {e3['pauli']}"


# ---------------------------------------------------------------------------
# B4. HYPERGRAPH TEST
# ---------------------------------------------------------------------------
class TestB4Hypergraph:
    """Hyperedge representation after CNOT(0,1) and CZ(2,3)."""

    def test_g0_hyperedge(self):
        """Generator g0=X0X1 maps to hyperedge {0,1}."""
        sim = _make_sim_n4_cnot01_cz23()
        hg = build_tanner_hypergraph(sim)
        assert 0 in hg
        assert hg[0]['qubits'] == {0, 1}
        assert hg[0]['weight'] == 2

    def test_g2_hyperedge(self):
        """Generator g2=X2Z3 maps to hyperedge {2,3}."""
        sim = _make_sim_n4_cnot01_cz23()
        hg = build_tanner_hypergraph(sim)
        assert 2 in hg
        assert hg[2]['qubits'] == {2, 3}
        assert hg[2]['weight'] == 2

    def test_g0_paulis(self):
        """g0=X0X1: both paulis are 'X'."""
        sim = _make_sim_n4_cnot01_cz23()
        hg = build_tanner_hypergraph(sim)
        assert hg[0]['paulis'] == {0: 'X', 1: 'X'}

    def test_g2_paulis(self):
        """g2=X2Z3: qubit 2 is 'X', qubit 3 is 'Z'."""
        sim = _make_sim_n4_cnot01_cz23()
        hg = build_tanner_hypergraph(sim)
        assert hg[2]['paulis'] == {2: 'X', 3: 'Z'}

    def test_g3_paulis(self):
        """g3=Z2X3: qubit 2 is 'Z', qubit 3 is 'X'."""
        sim = _make_sim_n4_cnot01_cz23()
        hg = build_tanner_hypergraph(sim)
        assert hg[3]['paulis'] == {2: 'Z', 3: 'X'}

    def test_weight1_generators_unchanged(self):
        """Generators untouched by gates still have weight 1."""
        sim = _make_sim_n4_cnot01_cz23()
        hg = build_tanner_hypergraph(sim)
        # g1 (destab q1) unaffected by CZ(2,3) but affected by CNOT(0,1)?
        # CNOT(0,1): g1 = X1 stays X1 (control unchanged), weight 1
        assert hg[1]['weight'] == 1
        assert hg[1]['qubits'] == {1}

    def test_hypergraph_total_keys(self):
        """All 2n=8 generators present in hypergraph."""
        sim = _make_sim_n4_cnot01_cz23()
        hg = build_tanner_hypergraph(sim)
        assert len(hg) == 8

    def test_hypergraph_no_destabilizers(self):
        """include_destabilizers=False: only stabilizer keys (r>=n)."""
        sim = _make_sim_n4_cnot01_cz23()
        n = 4
        hg = build_tanner_hypergraph(sim, include_destabilizers=False)
        assert all(r >= n for r in hg.keys())
        assert len(hg) == n


# ---------------------------------------------------------------------------
# B5. DEGREE ANNOTATION TEST
# ---------------------------------------------------------------------------
class TestB5DegreeAnnotation:
    """For every node, networkx degree matches simulator inv_len / supp_len."""

    def _check_degrees(self, sim):
        G = build_tanner_graph(sim)
        n = sim.n
        N = sim.N
        for q in range(n):
            expected = int(sim.inv_len[q])
            actual = G.degree(('v', q))
            assert actual == expected, (
                f"Variable v{q}: degree {actual} != inv_len {expected}")
        for r in range(N):
            expected = int(sim.supp_len[r])
            actual = G.degree(('c', r))
            assert actual == expected, (
                f"Check c{r}: degree {actual} != supp_len {expected}")

    def test_initial(self):
        self._check_degrees(_make_sim_n4_initial())

    def test_after_cnot(self):
        self._check_degrees(_make_sim_n4_cnot01())

    def test_after_cnot_cz(self):
        self._check_degrees(_make_sim_n4_cnot01_cz23())

    def test_n8_initial(self):
        self._check_degrees(SparseGF2(8))


# ---------------------------------------------------------------------------
# B6. VISUALIZATION SMOKE TEST
# ---------------------------------------------------------------------------
class TestB6VisualizationSmoke:
    """plot_tanner_graph returns Figure without exception."""

    def test_bipartite_n4(self):
        import matplotlib.figure
        sim = SparseGF2(4)
        fig = plot_tanner_graph(sim, mode='bipartite')
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_bipartite_n8(self):
        import matplotlib.figure
        sim = SparseGF2(8)
        fig = plot_tanner_graph(sim, mode='bipartite')
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_bipartite_n16(self):
        import matplotlib.figure
        sim = SparseGF2(16)
        fig = plot_tanner_graph(sim, mode='bipartite')
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_hypergraph_n4(self):
        import matplotlib.figure
        sim = SparseGF2(4)
        fig = plot_tanner_graph(sim, mode='hypergraph')
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_save_path(self):
        sim = SparseGF2(4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'tanner_test.png')
            fig = plot_tanner_graph(sim, mode='bipartite', save_path=path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
            matplotlib.pyplot.close(fig)

    def test_with_title_and_p(self):
        import matplotlib.figure
        sim = SparseGF2(4)
        fig = plot_tanner_graph(sim, mode='bipartite', title='Custom Title', p=0.15)
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_spring_mode(self):
        """Spring layout should work for small n (forced via mode='spring')."""
        import matplotlib.figure
        sim = SparseGF2(4)
        fig = plot_tanner_graph(sim, mode='spring')
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)


# ---------------------------------------------------------------------------
# B7. NODE TYPE TEST
# ---------------------------------------------------------------------------
class TestB7NodeType:
    """Stabilizer/destabilizer tagging and show_destabilizers=False filtering."""

    def test_stab_tagged_stabilizer(self):
        sim = _make_sim_n4_initial()
        G = build_tanner_graph(sim)
        n = 4
        for r in range(n, 2 * n):
            attrs = G.nodes[('c', r)]
            assert attrs['node_type'] == 'stabilizer', (
                f"c{r} (r>=n) should be tagged 'stabilizer', got '{attrs['node_type']}'")

    def test_destab_tagged_destabilizer(self):
        sim = _make_sim_n4_initial()
        G = build_tanner_graph(sim)
        n = 4
        for r in range(n):
            attrs = G.nodes[('c', r)]
            assert attrs['node_type'] == 'destabilizer', (
                f"c{r} (r<n) should be tagged 'destabilizer', got '{attrs['node_type']}'")

    def test_no_destabilizers_nodes(self):
        """With show_destabilizers=False (=> include_destabilizers=False):
        no destabilizer check nodes remain."""
        sim = _make_sim_n4_initial()
        G = build_tanner_graph(sim, include_destabilizers=False)
        n = 4
        for nd, attrs in G.nodes(data=True):
            if nd[0] == 'c':
                assert attrs['node_type'] != 'destabilizer', (
                    f"Node {nd} should not be a destabilizer")

    def test_no_destabilizers_edges(self):
        """With include_destabilizers=False: no edges to destabilizer check nodes."""
        sim = _make_sim_n4_initial()
        n = 4
        G = build_tanner_graph(sim, include_destabilizers=False)
        for u, v in G.edges():
            check_node = v if v[0] == 'c' else u
            r = check_node[1]
            assert r >= n, (
                f"Edge {u}-{v} connects to destabilizer check node c{r}")

    def test_no_destabilizers_reduced_count(self):
        """With include_destabilizers=False: only n check + n variable = 2n nodes."""
        sim = _make_sim_n4_initial()
        n = 4
        G = build_tanner_graph(sim, include_destabilizers=False)
        # n variable + n stabilizer check = 2n
        assert len(G.nodes()) == 2 * n

    def test_no_destabilizers_reduced_edges(self):
        """With include_destabilizers=False: edges only from stabilizers.
        Initial: n stab generators, each weight 1 => n edges."""
        sim = _make_sim_n4_initial()
        n = 4
        G = build_tanner_graph(sim, include_destabilizers=False)
        assert len(G.edges()) == n  # 4 stabilizers, each weight 1

    def test_no_destabilizers_degree_attribute_correct(self):
        """Degree attribute matches actual graph degree when destabs excluded."""
        sim = _make_sim_n4_initial()
        n = 4
        G = build_tanner_graph(sim, include_destabilizers=False)
        for q in range(n):
            attr_deg = G.nodes[('v', q)]['degree']
            graph_deg = G.degree(('v', q))
            assert attr_deg == graph_deg, (
                f"v{q}: degree attr={attr_deg} != graph degree={graph_deg}")
            # Initial state: each qubit has 1 stab => stab-only degree = 1
            assert graph_deg == 1, (
                f"v{q}: expected degree 1 (stab only), got {graph_deg}")

    def test_no_destabilizers_degree_after_gates(self):
        """After CNOT(0,1): g5=Z0Z1 spreads Z. Stab-only degree for q0,q1 = 2."""
        sim = _make_sim_n4_cnot01()
        n = 4
        G = build_tanner_graph(sim, include_destabilizers=False)
        for q in range(n):
            attr_deg = G.nodes[('v', q)]['degree']
            graph_deg = G.degree(('v', q))
            assert attr_deg == graph_deg

    def test_no_destabilizers_visualization_smoke(self):
        """plot_tanner_graph with show_destabilizers=False must not crash."""
        import matplotlib.figure
        sim = _make_sim_n4_cnot01_cz23()
        fig = plot_tanner_graph(sim, mode='bipartite', show_destabilizers=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)
