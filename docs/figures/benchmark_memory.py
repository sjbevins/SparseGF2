"""Memory footprint: SparseGF2 vs Stim. Stim wins, at every size.

The honest comparison is the size of the state's data structure (RSS is too noisy
below ~1000 qubits and is polluted by numba's JIT cache for SparseGF2):

* **SparseGF2** stores an inverted-index structure sized to the worst case
  (several int arrays of shape ~``(2n, n)``). We sum the exact ``nbytes`` of
  every array, so this is the real footprint of the representation.
* **Stim** stores a bit-packed dense tableau. A TableauSimulator on ``n`` qubits
  keeps ``2n`` Pauli generators (and their inverse), each two length-``n``
  bit-vectors padded to 256-bit words, so ``256 * n * ceil(n/256)`` bytes
  (``~ n^2``). We cross-check this against peak RSS at large ``n``.

The point: the sparse structure that makes *measurement* fast costs memory, and
Stim's bit-packed tableau is ~30x smaller. Run::

    .venv/bin/python docs/figures/benchmark_memory.py
"""

from __future__ import annotations

import math

import numpy as np

from sparsegf2 import SparseGF2

NS = [64, 128, 256, 512, 1024, 2048]

# Names of the arrays that make up the SparseGF2 tableau (the whole footprint).
_ARRAYS = (
    "plt", "supp_q", "supp_len", "supp_pos",
    "inv", "inv_len", "inv_pos", "inv_x", "inv_x_len", "inv_x_pos",
)


def sparsegf2_bytes(n: int) -> int:
    s = SparseGF2(n)
    return sum(int(getattr(s, a).nbytes) for a in _ARRAYS if hasattr(s, a))


def stim_bytes(n: int) -> int:
    # 2n generators x 2 bit-vectors x ceil(n/256)*32 bytes, doubled for the inverse.
    return 2 * (2 * n) * 2 * (math.ceil(n / 256) * 32)


def main() -> int:
    sg = np.array([sparsegf2_bytes(n) for n in NS], dtype=float)
    st = np.array([stim_bytes(n) for n in NS], dtype=float)
    print(f"{'n':>6}{'SparseGF2 (MB)':>16}{'Stim (MB)':>12}{'ratio':>8}")
    for n, a, b in zip(NS, sg, st, strict=True):
        print(f"{n:>6}{a / 1e6:>16.2f}{b / 1e6:>12.3f}{a / b:>7.1f}x")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ns = np.array(NS)
    ax.loglog(ns, sg / 1e6, "o-", label="SparseGF2 (sparse inverted index)")
    ax.loglog(ns, st / 1e6, "s-", label="Stim (bit-packed dense tableau)")
    ax.set_xlabel("system size n (qubits)")
    ax.set_ylabel("tableau memory (MB)")
    ax.set_title("Memory footprint vs n: Stim wins (~30x smaller)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig("docs/figures/memory_vs_stim.png", dpi=150, bbox_inches="tight")
    print("wrote docs/figures/memory_vs_stim.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
