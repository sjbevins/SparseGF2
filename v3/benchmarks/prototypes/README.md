# Layout prototypes (evidence for DESIGN.md section 5.1)

Throwaway kernels written during the audit to choose v3's data layout. They are
**not** part of the package and will be deleted once milestone 5 publishes the
real benchmarks. Each compares layouts on identical brickwork circuits (random
two-qubit Cliffords on a ring, Bernoulli-p Z measurements), driving every kernel
from one compiled Numba loop so Python overhead is excluded, and asserts that
all of them reach the same stabilizer group (and Stim, where included).

They import the v2 package from `../../../src`, so run them from a checkout of
this repository with numba and stim installed:

```sh
python v3/benchmarks/prototypes/layouts_v2_vs_dual.py          # v2 kernel vs dual layout, n = 128..1024 (slow: ~15 min)
python v3/benchmarks/prototypes/layouts_column_and_stim.py 128 256 512   # adds column layout and Stim do_circuit
python v3/benchmarks/prototypes/large_n.py 2048                # area-law regime at large n
```

Layouts: **v2** = existing kernels (2n rows, byte table + inverted indices);
**dual** = bit-packed rows and columns, stabilizer rows only; **column** =
bit-packed columns only, stabilizer rows only. All prototypes are phase-free.
