"""Prototype: compare kernel data layouts on a compiled brickwork MIPT loop.

A: old SparseGF2 kernels (2n rows, dense PLT + inverted indices), driven from one njit loop.
B: new "dual bit-packed" layout: row-major x/z bit planes + column-major x/z bit planes,
   stabilizer rows only (phase-free), driven from one njit loop.
"""
import sys
from pathlib import Path
import time

import numpy as np
from numba import njit

REPO = Path(__file__).resolve().parents[3]  # repository root (holds the v2 package in src/)
sys.path.insert(0, str(REPO / "src"))
from sparsegf2 import SparseGF2  # noqa: E402
from sparsegf2.core import numba_kernels as nk  # noqa: E402
from sparsegf2.core.linalg_gf2 import gf2_rref  # noqa: E402
from sparsegf2.core.sparse_tableau import _build_2q_lut  # noqa: E402
from sparsegf2.core.symplectic import enumerate_sp4  # noqa: E402

LUTS = np.stack([_build_2q_lut(S) for S in enumerate_sp4()]).astype(np.uint8)  # (720, 16)


def make_circuit(n, p, depth, seed):
    rng = np.random.default_rng(seed)
    pairs, gidx, gstart = [], [], [0]
    mq, mstart = [], [0]
    for t in range(depth):
        off = t % 2
        for i in range(off, n - 1 + off, 2):
            pairs.append((i % n, (i + 1) % n))
            gidx.append(rng.integers(720))
        gstart.append(len(pairs))
        for q in range(n):
            if rng.random() < p:
                mq.append(q)
        mstart.append(len(mq))
    return (np.array(pairs, np.int64).reshape(-1, 2), np.array(gidx, np.int64),
            np.array(gstart, np.int64), np.array(mq, np.int64), np.array(mstart, np.int64))


# ---------------------------------------------------------------- A: old kernels
@njit(cache=True)
def run_old(plt, supp_q, supp_len, supp_pos, inv, inv_len, inv_pos, inv_x, inv_x_len, inv_x_pos,
            active_buf, others_buf, pairs, gidx, gstart, mq, mstart, luts, minw):
    nrand = 0
    for t in range(gstart.shape[0] - 1):
        for g in range(gstart[t], gstart[t + 1]):
            nk.apply_2q_kernel_jit(plt, supp_q, supp_len, supp_pos, inv, inv_len, inv_pos,
                                   inv_x, inv_x_len, inv_x_pos, pairs[g, 0], pairs[g, 1],
                                   luts[gidx[g]], active_buf)
        for m in range(mstart[t], mstart[t + 1]):
            nrand += nk.measure_z_kernel_jit(plt, supp_q, supp_len, supp_pos, inv, inv_len,
                                             inv_pos, inv_x, inv_x_len, inv_x_pos, mq[m], minw,
                                             others_buf)
    return nrand


def old_sim(n, circ, minw=True):
    s = SparseGF2(n, use_numba=True, pivot_mode="min_weight" if minw else "first")
    t0 = time.perf_counter()
    nr = run_old(s.plt, s.supp_q, s._supp_len, s.supp_pos, s.inv, s.inv_len, s.inv_pos, s.inv_x,
                 s.inv_x_len, s.inv_x_pos, s._active_buf, s._others_buf, *circ, LUTS, minw)
    return time.perf_counter() - t0, nr, s


# ---------------------------------------------------------------- B: dual bit-packed
U1 = np.uint64(1)


@njit(cache=True, inline="always")
def _ctz(x):
    # count trailing zeros of nonzero uint64
    c = 0
    while (x & np.uint64(1)) == 0:
        x >>= np.uint64(1)
        c += 1
    return c


@njit(cache=True)
def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True)
def gate2(rx, rz, cx, cz, a, b, lut, rows_buf):
    # collect rows touching a or b from column planes
    nrw = cx.shape[1]
    cnt = 0
    for w in range(nrw):
        m = cx[a, w] | cz[a, w] | cx[b, w] | cz[b, w]
        while m:
            low = m & (~m + np.uint64(1))
            rows_buf[cnt] = (w << 6) + _ctz(low)
            cnt += 1
            m ^= low
    wa, ba = a >> 6, np.uint64(a & 63)
    wb, bb = b >> 6, np.uint64(b & 63)
    for k in range(cnt):
        r = rows_buf[k]
        xa = (rx[r, wa] >> ba) & U1
        za = (rz[r, wa] >> ba) & U1
        xb = (rx[r, wb] >> bb) & U1
        zb = (rz[r, wb] >> bb) & U1
        idx = (((xa << U1) | za) << np.uint64(2)) | ((xb << U1) | zb)
        out = np.uint64(lut[idx])
        if out == idx:
            continue
        nxa = (out >> np.uint64(3)) & U1
        nza = (out >> np.uint64(2)) & U1
        nxb = (out >> U1) & U1
        nzb = out & U1
        rx[r, wa] = (rx[r, wa] & ~(U1 << ba)) | (nxa << ba)
        rz[r, wa] = (rz[r, wa] & ~(U1 << ba)) | (nza << ba)
        rx[r, wb] = (rx[r, wb] & ~(U1 << bb)) | (nxb << bb)
        rz[r, wb] = (rz[r, wb] & ~(U1 << bb)) | (nzb << bb)
        wr, br = r >> 6, np.uint64(r & 63)
        bit = U1 << br
        if nxa != xa:
            cx[a, wr] ^= bit
        if nza != za:
            cz[a, wr] ^= bit
        if nxb != xb:
            cx[b, wr] ^= bit
        if nzb != zb:
            cz[b, wr] ^= bit


@njit(cache=True)
def measure_z(rx, rz, cx, cz, q, minw, rows_buf, supp_buf, wsupp_buf):
    nrw = cx.shape[1]
    nw = rx.shape[1]
    # anticommuting rows = rows with x on q
    k = 0
    for w in range(nrw):
        m = cx[q, w]
        while m:
            low = m & (~m + np.uint64(1))
            rows_buf[k] = (w << 6) + _ctz(low)
            k += 1
            m ^= low
    if k == 0:
        return 0
    # pivot
    piv_i = 0
    if minw and k > 1:
        best = 1 << 30
        for i in range(k):
            r = rows_buf[i]
            wt = 0
            for w in range(nw):
                wt += popcount64(rx[r, w] | rz[r, w])
            if wt < best:
                best = wt
                piv_i = i
    p = rows_buf[piv_i]
    rows_buf[piv_i] = rows_buf[k - 1]
    k -= 1
    # pivot support: nonzero words and qubits
    nsw = 0
    ns = 0
    for w in range(nw):
        m = rx[p, w] | rz[p, w]
        if m:
            wsupp_buf[nsw] = w
            nsw += 1
            while m:
                low = m & (~m + np.uint64(1))
                supp_buf[ns] = (w << 6) + _ctz(low)
                ns += 1
                m ^= low
    # row-major update
    for i in range(k):
        r = rows_buf[i]
        for j in range(nsw):
            w = wsupp_buf[j]
            rx[r, w] ^= rx[p, w]
            rz[r, w] ^= rz[p, w]
    # column-major update
    if k * 2 < nrw * 3:
        for j in range(ns):
            c = supp_buf[j]
            wc, bc = c >> 6, np.uint64(c & 63)
            px = (rx[p, wc] >> bc) & U1
            pz = (rz[p, wc] >> bc) & U1
            for i in range(k):
                r = rows_buf[i]
                bit = U1 << np.uint64(r & 63)
                if px:
                    cx[c, r >> 6] ^= bit
                if pz:
                    cz[c, r >> 6] ^= bit
    else:
        # word-parallel: build anti mask (excluding pivot) once
        mask = np.zeros(nrw, np.uint64)
        for i in range(k):
            r = rows_buf[i]
            mask[r >> 6] |= U1 << np.uint64(r & 63)
        for j in range(ns):
            c = supp_buf[j]
            wc, bc = c >> 6, np.uint64(c & 63)
            px = (rx[p, wc] >> bc) & U1
            pz = (rz[p, wc] >> bc) & U1
            if px:
                for w in range(nrw):
                    cx[c, w] ^= mask[w]
            if pz:
                for w in range(nrw):
                    cz[c, w] ^= mask[w]
    # replace pivot by Z_q
    pw, pb = p >> 6, ~(U1 << np.uint64(p & 63))
    for j in range(ns):
        c = supp_buf[j]
        cx[c, pw] &= pb
        cz[c, pw] &= pb
    for j in range(nsw):
        w = wsupp_buf[j]
        rx[p, w] = 0
        rz[p, w] = 0
    rz[p, q >> 6] |= U1 << np.uint64(q & 63)
    cz[q, pw] |= U1 << np.uint64(p & 63)
    return 1


@njit(cache=True)
def run_new(rx, rz, cx, cz, pairs, gidx, gstart, mq, mstart, luts, minw):
    R = rx.shape[0]
    n = cx.shape[0]
    rows_buf = np.zeros(R, np.int64)
    supp_buf = np.zeros(n, np.int64)
    wsupp_buf = np.zeros(rx.shape[1], np.int64)
    nrand = 0
    for t in range(gstart.shape[0] - 1):
        for g in range(gstart[t], gstart[t + 1]):
            gate2(rx, rz, cx, cz, pairs[g, 0], pairs[g, 1], luts[gidx[g]], rows_buf)
        for m in range(mstart[t], mstart[t + 1]):
            nrand += measure_z(rx, rz, cx, cz, mq[m], minw, rows_buf, supp_buf, wsupp_buf)
    return nrand


def new_sim(n, circ, minw=True):
    W = (n + 63) // 64
    rx = np.zeros((n, W), np.uint64)
    rz = np.zeros((n, W), np.uint64)
    cx = np.zeros((n, W), np.uint64)
    cz = np.zeros((n, W), np.uint64)
    for i in range(n):
        rz[i, i >> 6] |= np.uint64(1) << np.uint64(i & 63)
        cz[i, i >> 6] |= np.uint64(1) << np.uint64(i & 63)
    t0 = time.perf_counter()
    nr = run_new(rx, rz, cx, cz, *circ, LUTS, minw)
    return time.perf_counter() - t0, nr, (rx, rz, cx, cz)


def new_stab_matrix(n, st):
    rx, rz, cx, cz = st
    out = np.zeros((n, 2 * n), np.uint8)
    for r in range(n):
        for q in range(n):
            out[r, q] = (int(rx[r, q >> 6]) >> (q & 63)) & 1
            out[r, n + q] = (int(rz[r, q >> 6]) >> (q & 63)) & 1
    # check column planes agree
    for q in range(n):
        for r in range(n):
            assert ((int(cx[q, r >> 6]) >> (r & 63)) & 1) == out[r, q]
            assert ((int(cz[q, r >> 6]) >> (r & 63)) & 1) == out[r, n + q]
    return out


if __name__ == "__main__":
    # correctness
    for n, p in ((20, 0.1), (64, 0.2), (130, 0.15)):
        circ = make_circuit(n, p, 3 * n, seed=n)
        _, nr_old, s = old_sim(n, circ)
        _, nr_new, st = new_sim(n, circ)
        assert nr_old == nr_new, (nr_old, nr_new)
        assert np.array_equal(gf2_rref(new_stab_matrix(n, st)), s.canonical_form())
    print("correctness OK")
    print(f"{'n':>5} {'p':>5} {'depth':>6} | {'old(2n rows)':>12} {'new(n rows)':>12} {'speedup':>8}")
    for n in (128, 256, 512, 1024):
        for p in (0.05, 0.1, 0.16, 0.2, 0.3):
            depth = 4 * n
            circ = make_circuit(n, p, depth, seed=1)
            t_old, nr1, _ = old_sim(n, circ)
            t_new, nr2, _ = new_sim(n, circ)
            assert nr1 == nr2
            print(f"{n:>5} {p:>5} {depth:>6} | {t_old:>11.3f}s {t_new:>11.3f}s {t_old / t_new:>7.1f}x")
