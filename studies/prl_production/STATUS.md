# PRL production status

Static snapshot: **2026-08-11 (America/New_York)**.

- **Production state:** paused after the runner reported a failure
- **Run ID:** `f76def10804a67a6`
- **Simulator review:** complete and published
- **Smoke validation:** complete
- **Points complete:** 8,537 / 37,400
- **Trajectories complete:** 4,270,300 / 18,700,000
- **Observed purification events:** 3,604,672

| n | trajectories complete | points complete |
|---:|---:|---:|
| 32 | 534,500 / 2,337,500 | 1,069 / 4,675 |
| 48 | 534,500 / 2,337,500 | 1,069 / 4,675 |
| 64 | 534,425 / 2,337,500 | 1,068 / 4,675 |
| 96 | 534,125 / 2,337,500 | 1,068 / 4,675 |
| 128 | 533,500 / 2,337,500 | 1,067 / 4,675 |
| 160 | 533,375 / 2,337,500 | 1,066 / 4,675 |
| 192 | 533,200 / 2,337,500 | 1,066 / 4,675 |
| 256 | 532,675 / 2,337,500 | 1,064 / 4,675 |

The campaign uses Watts--Strogatz rewiring of `C(n,2)`, exactly `2n` edges
and mean degree four, with `T_max=8n` and `S(R)` evaluated after every measured
layer. Completed point files were written atomically and remain valid. An
identical launch resumes incomplete graph indices rather than repeating
completed trajectories; inspect the ignored local runner log before resuming
to determine the failure cause.

This file is a version-controlled snapshot, not a live monitor. Generated
data, runtime manifests, and logs remain local. See
[`PROVENANCE.md`](PROVENANCE.md) for the immutable source fingerprint and
published commit required to resume this run.
