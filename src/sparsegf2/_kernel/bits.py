"""Bit-twiddling primitives for 64-bit words, compiled to single CPU instructions.

``popcount`` and ``ctz`` (count trailing zeros) map to LLVM's ``ctpop`` and
``cttz`` intrinsics, which compile to POPCNT/TZCNT on x86-64 and the
equivalents on ARM.
"""

from __future__ import annotations

import llvmlite.ir as ir
import numba
import numpy as np
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic

ONE = np.uint64(1)
ZERO = np.uint64(0)
ALL = np.uint64(0xFFFFFFFFFFFFFFFF)


@intrinsic
def popcount(typingctx, x):  # pragma: no cover (compiled code)
    """Number of set bits in a uint64."""
    sig = types.uint64(types.uint64)

    def codegen(context, builder, signature, args):
        i64 = ir.IntType(64)
        fn = cgutils.get_or_insert_function(
            builder.module, ir.FunctionType(i64, [i64]), "llvm.ctpop.i64"
        )
        return builder.call(fn, args)

    return sig, codegen


@intrinsic
def ctz(typingctx, x):  # pragma: no cover (compiled code)
    """Index of the lowest set bit of a uint64 (undefined for 0; callers check)."""
    sig = types.uint64(types.uint64)

    def codegen(context, builder, signature, args):
        i64 = ir.IntType(64)
        i1 = ir.IntType(1)
        fn = cgutils.get_or_insert_function(
            builder.module, ir.FunctionType(i64, [i64, i1]), "llvm.cttz.i64"
        )
        return builder.call(fn, [args[0], ir.Constant(i1, 1)])

    return sig, codegen


if numba.config.DISABLE_JIT:
    # Intrinsics cannot run as plain Python. With NUMBA_DISABLE_JIT=1 (used to trace the
    # kernels for coverage) the kernels run as Python, so these stand in for them.

    def popcount(x):  # noqa: F811
        """Number of set bits in a uint64."""
        return np.uint64(int(x).bit_count())

    def ctz(x):  # noqa: F811
        """Index of the lowest set bit of a uint64; 0 is an error here, as callers check."""
        x = int(x)
        if x == 0:
            raise ValueError("ctz(0) is undefined")
        return np.uint64((x & -x).bit_length() - 1)


@njit(cache=True, inline="always")
def prefix_parity_exclusive(v):
    """Bit l of the result is the XOR of bits 0..l-1 of ``v``."""
    t = v << ONE
    t ^= t << ONE
    t ^= t << np.uint64(2)
    t ^= t << np.uint64(4)
    t ^= t << np.uint64(8)
    t ^= t << np.uint64(16)
    t ^= t << np.uint64(32)
    return t
