# Circuits in Stim's language

The `circuit` stack runs Stim-language text once or for many shots. Pass the text as `circuit`, or a file as `circuit_file`.

- The dialect is Stim's: gates, measurements (M, MX, MPP, ...), resets, noise, detectors, observables, TICK and REPEAT. `help_list(what="gates")` and `help_list(what="instructions")` give the exact names.
- One extension: `CLIFFORD_2Q(i) a b` applies the two-qubit Clifford number `i` in [0, 11520). A uniformly random `i` gives a uniformly random Clifford.

Tools:

1. `circuit_inspect` first. It returns counts and an instruction histogram, or every bad line with its line number (including `rec[-k]` lookbacks that point before the first measurement, and circuits too large to unroll).
2. `circuit_run` runs it. With `shots=1` you get the measurement record as a bitstring. With more shots you get per-measurement means and detector and observable flip rates. `observables` (call strings) are evaluated on the final state; `observe_ticks=true` also evaluates them after every TICK, and `tick_stride=k` after every k-th tick only.
3. `circuit_convert` with `to=stim` expands CLIFFORD_2Q into Stim gates, so Stim can read the circuit; `to=canonical` normalises the text.
4. `circuit_crosscheck` runs the circuit in SparseGF2 and in Stim and compares the stabilizer groups at the ticks, and with `compare=measurements` every outcome, detector and observable. It reports `match` or the first mismatch. It works for noise-free circuits only, and needs Stim installed.

Example: `{"circuit": "H 0\nCX 0 1\nTICK\nM 0 1", "shots": 1000, "seed": 1}` gives means near 0.5 for both measurements, which are always equal.
