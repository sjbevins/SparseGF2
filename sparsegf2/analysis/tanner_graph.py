"""Tanner graph construction and visualization for SparseGF2 tableaux.

Builds bipartite graphs G = (V U C, E) where V = variable nodes (qubits),
C = check nodes (generators), E = edges for nontrivial Pauli entries.
Also supports hypergraph representation where each generator is a hyperedge.

Requires networkx for graph construction (soft dependency).
Optionally uses pyvis for interactive HTML visualization.
"""
import numpy as np
from sparsegf2.analysis._numba_kernels import _ensure_sparse_indices

# Pauli encoding: 0=I, 1=Z, 2=X, 3=Y
_PAULI_NAMES = {1: 'Z', 2: 'X', 3: 'Y'}
_PAULI_EDGE_COLORS = {'X': '#d62728', 'Z': '#1f77b4', 'Y': '#9467bd'}


def build_tanner_graph(sim, include_destabilizers=True):
    """Build a bipartite Tanner graph from the current tableau state.

    G = (V U C, E) where:
      V = {('v', q) : q = 0,...,n-1}     -- variable nodes (qubits)
      C = {('c', r) : r in range}        -- check nodes (generators)
      E = {(('v',q), ('c',r)) : plt[r,q] != 0}

    |E| = weight_mass = n * abar = 2n * wbar exactly.

    Node attributes:
      Variable: bipartite=0, node_type='variable', degree=inv_len[q], qubit=q
      Check:    bipartite=1, node_type='stabilizer'|'destabilizer',
                weight=supp_len[r], generator=r

    Edge attributes:
      pauli='X'|'Z'|'Y', pauli_val=2|1|3

    Parameters
    ----------
    sim : SparseGF2
    include_destabilizers : bool
        If False, only stabilizer check nodes (r >= n) are included.

    Returns
    -------
    networkx.Graph
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "networkx is required for Tanner graph construction. "
            "Install with: pip install networkx")

    _ensure_sparse_indices(sim)
    n = sim.n
    N = sim.N

    G = nx.Graph()

    # Variable nodes (degree set after edges are added)
    for q in range(n):
        G.add_node(('v', q), bipartite=0, node_type='variable',
                   degree=0, qubit=q)

    # Check nodes + edges (reuse supp_q to avoid scanning plt)
    r_start = 0 if include_destabilizers else n
    for r in range(r_start, N):
        node_type = 'stabilizer' if r >= n else 'destabilizer'
        G.add_node(('c', r), bipartite=1, node_type=node_type,
                   weight=int(sim.supp_len[r]), generator=r)

        for idx in range(sim.supp_len[r]):
            q = int(sim.supp_q[r, idx])
            pv = int(sim.plt[r, q])
            G.add_edge(('v', q), ('c', r),
                       pauli=_PAULI_NAMES.get(pv, '?'), pauli_val=pv)

    # Sync degree attribute to actual graph degree (correct for both modes)
    for q in range(n):
        G.nodes[('v', q)]['degree'] = G.degree(('v', q))

    return G


def build_tanner_hypergraph(sim, include_destabilizers=True):
    """Build a hypergraph representation of the stabilizer tableau.

    Each generator r corresponds to a hyperedge connecting all qubits
    in its support supp_q[r].

    Parameters
    ----------
    sim : SparseGF2
    include_destabilizers : bool
        If False, only stabilizer generators (r >= n).

    Returns
    -------
    dict
        {r: {'qubits': set, 'paulis': {q: str}, 'type': str, 'weight': int}}
    """
    _ensure_sparse_indices(sim)
    n = sim.n
    N = sim.N
    r_start = 0 if include_destabilizers else n

    hypergraph = {}
    for r in range(r_start, N):
        qubits = set()
        paulis = {}
        for idx in range(sim.supp_len[r]):
            q = int(sim.supp_q[r, idx])
            qubits.add(q)
            paulis[q] = _PAULI_NAMES.get(int(sim.plt[r, q]), '?')

        hypergraph[r] = {
            'qubits': qubits,
            'paulis': paulis,
            'type': 'stabilizer' if r >= n else 'destabilizer',
            'weight': int(sim.supp_len[r]),
        }
    return hypergraph


def plot_tanner_graph(
    sim,
    mode='bipartite',
    show_destabilizers=True,
    pauli_edge_colors=True,
    figsize=None,
    ax=None,
    title=None,
    p=None,
    save_path=None,
    dpi=150,
):
    """Visualize the Tanner graph of the current tableau state.

    Parameters
    ----------
    sim : SparseGF2
    mode : str
        'bipartite', 'spring', or 'hypergraph'.
        'bipartite' auto-switches to 'spring' for n > 32.
        'hypergraph' falls back to 'bipartite' for n > 24.
    show_destabilizers : bool
        Include destabilizer check nodes and their edges.
    pauli_edge_colors : bool
        Color edges by Pauli type (X=red, Z=blue, Y=purple).
    figsize : tuple, optional
        Auto-sized based on n if None.
    ax : matplotlib.axes.Axes, optional
        Inject axes for subplots. Creates new figure if None.
    title : str, optional
        Auto-generated with n, abar, wbar if None.
    p : float, optional
        Measurement rate metadata for title.
    save_path : str, optional
        Save figure to this path.
    dpi : int
        DPI for saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    import matplotlib.pyplot as plt_mod
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "networkx is required for Tanner graph visualization. "
            "Install with: pip install networkx")

    n = sim.n

    # Auto-adjust mode for large n
    if mode == 'hypergraph' and n > 24:
        mode = 'bipartite'
    if mode == 'bipartite' and n > 32:
        mode = 'spring'

    # Hypergraph mode is drawn separately
    if mode == 'hypergraph':
        return _plot_hypergraph(sim, show_destabilizers, pauli_edge_colors,
                                figsize, ax, title, p, save_path, dpi)

    G = build_tanner_graph(sim, include_destabilizers=show_destabilizers)

    if figsize is None:
        figsize = (max(10, n * 0.3), max(6, n * 0.2))

    if ax is None:
        fig, ax = plt_mod.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Layout
    var_nodes = sorted([nd for nd in G if nd[0] == 'v'], key=lambda x: x[1])
    check_nodes = sorted([nd for nd in G if nd[0] == 'c'], key=lambda x: x[1])

    if mode == 'bipartite':
        pos = {}
        for i, nd in enumerate(var_nodes):
            pos[nd] = (i / max(len(var_nodes) - 1, 1), 1.0)
        for i, nd in enumerate(check_nodes):
            pos[nd] = (i / max(len(check_nodes) - 1, 1), 0.0)
        layout_label = "bipartite layout"
    else:  # spring
        pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(max(len(G), 1)))
        layout_label = "spring layout"

    # Edges
    edge_alpha = max(0.1, min(0.8, 20.0 / max(len(G.edges()), 1)))
    for u, v, data in G.edges(data=True):
        pauli = data.get('pauli', '?')
        color = _PAULI_EDGE_COLORS.get(pauli, '#999999') if pauli_edge_colors else '#cccccc'
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=color, alpha=edge_alpha, linewidth=0.8, zorder=1)

    # Variable nodes: color by degree via viridis
    if var_nodes:
        degs = [G.nodes[nd]['degree'] for nd in var_nodes]
        max_d = max(degs) if degs else 1
        min_d = min(degs) if degs else 1
        norm_v = plt_mod.Normalize(min_d, max(max_d, min_d + 1))
        sizes = [max(30, 20 + 80 * (d / max(max_d, 1))) for d in degs]
        colors = [cm.viridis(norm_v(d)) for d in degs]
        xs = [pos[nd][0] for nd in var_nodes]
        ys = [pos[nd][1] for nd in var_nodes]
        ax.scatter(xs, ys, s=sizes, c=colors, marker='o',
                   edgecolors='black', linewidth=0.5, zorder=3)
        if n <= 16:
            for nd in var_nodes:
                ax.annotate(f"q{nd[1]}", pos[nd], fontsize=6,
                            ha='center', va='center', zorder=4)

    # Check nodes: stabilizers=square, destabilizers=circle, color by weight via plasma
    if check_nodes:
        all_w = [G.nodes[nd]['weight'] for nd in check_nodes]
        max_w = max(all_w) if all_w else 1
        min_w = min(all_w) if all_w else 0
        norm_w = plt_mod.Normalize(min_w, max(max_w, min_w + 1))

        for node_type, marker in [('stabilizer', 's'), ('destabilizer', 'o')]:
            subset = [nd for nd in check_nodes
                      if G.nodes[nd]['node_type'] == node_type]
            if not subset:
                continue
            ws = [G.nodes[nd]['weight'] for nd in subset]
            sizes = [max(20, 15 + 60 * (w / max(max_w, 1))) for w in ws]
            colors = [cm.plasma(norm_w(w)) for w in ws]
            xs = [pos[nd][0] for nd in subset]
            ys = [pos[nd][1] for nd in subset]
            ax.scatter(xs, ys, s=sizes, c=colors, marker=marker,
                       edgecolors='black', linewidth=0.5, zorder=3)
            if n <= 16:
                for nd in subset:
                    r = nd[1]
                    label = f"s{r - n}" if r >= n else f"d{r}"
                    ax.annotate(label, pos[nd], fontsize=5,
                                ha='center', va='center', zorder=4)

    # Legend
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=8, label='Variable (qubit)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=8, label='Check (stabilizer)'),
    ]
    if show_destabilizers:
        handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                   markersize=6, label='Check (destabilizer)'))
    if pauli_edge_colors:
        handles.extend([
            Line2D([0], [0], color='#d62728', lw=2, label='X'),
            Line2D([0], [0], color='#1f77b4', lw=2, label='Z'),
            Line2D([0], [0], color='#9467bd', lw=2, label='Y'),
        ])
    ax.legend(handles=handles, loc='upper right', fontsize=7, framealpha=0.8)

    # Title (lightweight O(n) computation, avoids full compute_weight_stats)
    if title is None:
        _ensure_sparse_indices(sim)
        abar = float(np.sum(sim.inv_len[:n])) / n
        wbar = float(np.sum(sim.supp_len[:sim.N])) / sim.N
        title = f"Tanner Graph (n={n}, \u0101={abar:.2f}, \u0175={wbar:.2f}"
        if p is not None:
            title += f", p={p:.3f}"
        title += ")"
    ax.set_title(title, fontsize=10)
    ax.text(0.5, -0.02, layout_label, transform=ax.transAxes,
            ha='center', fontsize=8, color='gray')
    ax.set_axis_off()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig


def _plot_hypergraph(sim, show_destabilizers, pauli_edge_colors,
                     figsize, ax, title, p, save_path, dpi):
    """Draw the hypergraph visualization with convex hulls on a circular layout."""
    import matplotlib.pyplot as plt_mod
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D

    _ensure_sparse_indices(sim)
    n = sim.n
    N = sim.N

    if figsize is None:
        figsize = (max(8, n * 0.5), max(6, n * 0.4))
    if ax is None:
        fig, ax = plt_mod.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Qubits on a circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    qpos = {q: (np.cos(angles[q]), np.sin(angles[q])) for q in range(n)}

    # Draw generators as hulls/lines
    hull_colors = {'X': ('#d62728', 0.15), 'Z': ('#1f77b4', 0.15),
                   'Y': ('#9467bd', 0.15), 'mixed': ('#2ca02c', 0.10)}
    r_start = 0 if show_destabilizers else n

    for r in range(r_start, N):
        L = sim.supp_len[r]
        if L == 0:
            continue
        qs = [int(sim.supp_q[r, idx]) for idx in range(L)]
        ptypes = {_PAULI_NAMES.get(int(sim.plt[r, q]), '?') for q in qs}
        ptype = ptypes.pop() if len(ptypes) == 1 else 'mixed'
        color, alpha = hull_colors.get(ptype, ('#999999', 0.10))

        if L == 1:
            circle = plt_mod.Circle(qpos[qs[0]], 0.08, color=color,
                                    alpha=alpha + 0.15, zorder=2)
            ax.add_patch(circle)
        elif L == 2:
            pts = np.array([qpos[q] for q in qs])
            ax.plot(pts[:, 0], pts[:, 1], color=color,
                    alpha=alpha + 0.3, linewidth=3, zorder=2)
        else:
            pts = np.array([qpos[q] for q in qs])
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                poly = Polygon(hull_pts, closed=True, facecolor=color,
                               alpha=alpha, edgecolor=color, linewidth=1.5, zorder=2)
                ax.add_patch(poly)
            except Exception:
                # Fallback for collinear or degenerate points
                for i in range(len(qs)):
                    for j in range(i + 1, len(qs)):
                        ax.plot([qpos[qs[i]][0], qpos[qs[j]][0]],
                                [qpos[qs[i]][1], qpos[qs[j]][1]],
                                color=color, alpha=alpha + 0.2, linewidth=2, zorder=2)

    # Qubit nodes on top
    for q in range(n):
        ax.plot(*qpos[q], 'o', color='black', markersize=8, zorder=5)
        if n <= 16:
            ax.annotate(f"q{q}", qpos[q], fontsize=7, ha='center',
                        va='bottom', xytext=(0, 6), textcoords='offset points', zorder=6)

    # Legend
    handles = []
    if pauli_edge_colors:
        handles.extend([
            Line2D([0], [0], color='#d62728', lw=4, alpha=0.4, label='X hyperedge'),
            Line2D([0], [0], color='#1f77b4', lw=4, alpha=0.4, label='Z hyperedge'),
            Line2D([0], [0], color='#9467bd', lw=4, alpha=0.4, label='Y hyperedge'),
            Line2D([0], [0], color='#2ca02c', lw=4, alpha=0.4, label='Mixed hyperedge'),
        ])
    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=7, framealpha=0.8)

    if title is None:
        abar = float(np.sum(sim.inv_len[:n])) / n
        wbar = float(np.sum(sim.supp_len[:N])) / N
        title = f"Tanner Hypergraph (n={n}, \u0101={abar:.2f}, \u0175={wbar:.2f}"
        if p is not None:
            title += f", p={p:.3f}"
        title += ")"
    ax.set_title(title, fontsize=10)
    ax.text(0.5, -0.02, "hypergraph layout", transform=ax.transAxes,
            ha='center', fontsize=8, color='gray')

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig
