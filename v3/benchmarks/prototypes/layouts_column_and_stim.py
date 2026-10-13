"""Prototype C: column-major bit planes only (Stim-like gates), stabilizer rows only, phase-free.
Plus Stim driven at full speed (prebuilt stim.Circuit + do_circuit, batched M) for reference."""
import sys
from pathlib import Path
import time

import numpy as np
import stim
from numba import njit

REPO = Path(__file__).resolve().parents[3]  # repository root (holds the v2 package in src/)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from layouts_v2_vs_dual import LUTS, make_circuit, new_sim, old_sim  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
from sparsegf2.core.linalg_gf2 import gf2_rref  # noqa: E402
from sparsegf2.core.symplectic import enumerate_sp4  # noqa: E402

SP4 = enumerate_sp4().astype(np.uint8)  # (720, 4, 4)
U1 = np.uint64(1)
ALL = np.uint64(0xFFFFFFFFFFFFFFFF)
MASKS = np.where(SP4.astype(bool), ALL, np.uint64(0)).astype(np.uint64)  # (720,4,4)


@njit(cache=True)
def popc(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True)
def gate2_col(xc, zc, a, b, M):
    # (xa, xb, za, zb) @ S, word-parallel over rows
    for w in range(xc.shape[1]):
        v0 = xc[a, w]
        v1 = xc[b, w]
        v2 = zc[a, w]
        v3 = zc[b, w]
        xc[a, w] = (v0 & M[0, 0]) ^ (v1 & M[1, 0]) ^ (v2 & M[2, 0]) ^ (v3 & M[3, 0])
        xc[b, w] = (v0 & M[0, 1]) ^ (v1 & M[1, 1]) ^ (v2 & M[2, 1]) ^ (v3 & M[3, 1])
        zc[a, w] = (v0 & M[0, 2]) ^ (v1 & M[1, 2]) ^ (v2 & M[2, 2]) ^ (v3 & M[3, 2])
        zc[b, w] = (v0 & M[0, 3]) ^ (v1 & M[1, 3]) ^ (v2 & M[2, 3]) ^ (v3 & M[3, 3])


@njit(cache=True)
def measure_z_col(xc, zc, q, minw, supp_buf, mask):
    n = xc.shape[0]
    nrw = xc.shape[1]
    # anti mask
    k = 0
    first = -1
    for w in range(nrw):
        m = xc[q, w]
        mask[w] = m
        if m:
            k += popc(m)
            if first < 0:
                low = m & (~m + U1)
                c = 0
                while (low >> np.uint64(c)) != U1:
                    c += 1
                first = (w << 6) + c
    if k == 0:
        return 0
    p = first
    if minw and k > 1:
        best = 1 << 30
        for w in range(nrw):
            m = mask[w]
            while m:
                low = m & (~m + U1)
                c = 0
                while (low >> np.uint64(c)) != U1:
                    c += 1
                r = (w << 6) + c
                m ^= low
                rw, rb = r >> 6, np.uint64(r & 63)
                wt = 0
                for cc in range(n):
                    wt += int(((xc[cc, rw] | zc[cc, rw]) >> rb) & U1)
                    if wt >= best:
                        break
                if wt < best:
                    best = wt
                    p = r
    pw, pb = p >> 6, np.uint64(p & 63)
    mask[pw] &= ~(U1 << pb)
    # pivot support (probe every column)
    ns = 0
    for c in range(n):
        if ((xc[c, pw] | zc[c, pw]) >> pb) & U1:
            supp_buf[ns] = c
            ns += 1
    for j in range(ns):
        c = supp_buf[j]
        px = (xc[c, pw] >> pb) & U1
        pz = (zc[c, pw] >> pb) & U1
        if px:
            for w in range(nrw):
                xc[c, w] ^= mask[w]
        if pz:
            for w in range(nrw):
                zc[c, w] ^= mask[w]
        xc[c, pw] &= ~(U1 << pb)
        zc[c, pw] &= ~(U1 << pb)
    zc[q, pw] |= U1 << pb
    return 1


@njit(cache=True)
def run_col(xc, zc, pairs, gidx, gstart, mq, mstart, masks, minw):
    n = xc.shape[0]
    supp_buf = np.zeros(n, np.int64)
    mask = np.zeros(xc.shape[1], np.uint64)
    nrand = 0
    for t in range(gstart.shape[0] - 1):
        for g in range(gstart[t], gstart[t + 1]):
            gate2_col(xc, zc, pairs[g, 0], pairs[g, 1], masks[gidx[g]])
        for m in range(mstart[t], mstart[t + 1]):
            nrand += measure_z_col(xc, zc, mq[m], minw, supp_buf, mask)
    return nrand


def col_sim(n, circ, minw=True):
    RW = (n + 63) // 64
    xc = np.zeros((n, RW), np.uint64)
    zc = np.zeros((n, RW), np.uint64)
    for i in range(n):
        zc[i, i >> 6] |= np.uint64(1) << np.uint64(i & 63)
    t0 = time.perf_counter()
    nr = run_col(xc, zc, *circ, MASKS, minw)
    return time.perf_counter() - t0, nr, (xc, zc)


def col_stab_matrix(n, st):
    xc, zc = st
    out = np.zeros((n, 2 * n), np.uint8)
    for q in range(n):
        for r in range(n):
            out[r, q] = (int(xc[q, r >> 6]) >> (r & 63)) & 1
            out[r, n + q] = (int(zc[q, r >> 6]) >> (r & 63)) & 1
    return out


# ---- Stim at full speed
def _stim_gate_texts():
    out = []
    for S in SP4:
        # rows of S are images of (X_a, X_b, Z_a, Z_b) in [x_a x_b | z_a z_b] coords
        def ps(row):
            s = ""
            for q in range(2):
                x, z = row[q], row[2 + q]
                s += "_XZY"[x + 2 * z]
            return stim.PauliString(s)
        t = stim.Tableau.from_conjugated_generators(
            xs=[ps(S[0]), ps(S[1])], zs=[ps(S[2]), ps(S[3])])
        out.append(t.to_circuit("elimination"))
    return out


STIM_GATES = None


def stim_circuit(n, circ):
    global STIM_GATES
    if STIM_GATES is None:
        STIM_GATES = _stim_gate_texts()
    pairs, gidx, gstart, mq, mstart = circ
    c = stim.Circuit()
    for t in range(gstart.shape[0] - 1):
        for g in range(gstart[t], gstart[t + 1]):
            a, b = int(pairs[g, 0]), int(pairs[g, 1])
            sub = STIM_GATES[gidx[g]]
            # remap qubits 0->a, 1->b
            for inst in sub:
                tg = [a if tt.value == 0 else b for tt in inst.targets_copy()]
                c.append(inst.name, tg)
        qs = mq[mstart[t]:mstart[t + 1]].tolist()
        if qs:
            c.append("M", qs)
    return c


def stim_run(n, circ):
    c = stim_circuit(n, circ)
    s = stim.TableauSimulator(seed=0)
    s.set_num_qubits(n)
    t0 = time.perf_counter()
    s.do_circuit(c)
    return time.perf_counter() - t0, s


if __name__ == "__main__":
    for n, p in ((20, 0.1), (64, 0.2), (130, 0.15)):
        circ = make_circuit(n, p, 3 * n, seed=n)
        _, nr_old, s = old_sim(n, circ)
        for minw in (True, False):
            _, nr_c, st = col_sim(n, circ, minw)
            assert nr_old == nr_c
            assert np.array_equal(gf2_rref(col_stab_matrix(n, st)), s.canonical_form())
        _, ss = stim_run(n, circ)
        stabs = ss.canonical_stabilizers()
        mat = np.zeros((n, 2 * n), np.uint8)
        for i, P in enumerate(stabs):
            xs, zs = P.to_numpy()
            mat[i, :n] = xs
            mat[i, n:] = zs
        assert np.array_equal(gf2_rref(mat), s.canonical_form())
    print("correctness OK (old == col-only == stim)", flush=True)
    sizes = [int(a) for a in sys.argv[1:]] or [128, 256, 512]
    print(f"{'n':>5} {'p':>5} | {'old':>8} {'dual':>8} {'dual1st':>8} {'col':>8} {'col1st':>8} {'stim':>8}", flush=True)
    for n in sizes:
        for p in (0.05, 0.1, 0.16, 0.2, 0.3):
            depth = 4 * n
            circ = make_circuit(n, p, depth, seed=1)
            r = {}
            r["old"] = old_sim(n, circ)[0]
            r["dual"] = new_sim(n, circ, True)[0]
            r["dual1st"] = new_sim(n, circ, False)[0]
            r["col"] = col_sim(n, circ, True)[0]
            r["col1st"] = col_sim(n, circ, False)[0]
            r["stim"] = stim_run(n, circ)[0] if n <= 512 else float("nan")
            print(f"{n:>5} {p:>5} | " + " ".join(f"{r[k]:>7.3f}s" for k in r), flush=True)
