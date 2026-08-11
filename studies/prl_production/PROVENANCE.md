# Production provenance

## Active single-reference run

- Run ID: `f76def10804a67a6`
- Source fingerprint: `65a9e3e534a37f136ac7ee2b18fbcb0acbb83214b2e61603b195c67998695966`
- Published source commit: `eef62e26e8ea248c7c6e33ecacdff3a1180920c9`
- Public repository: `https://github.com/sjbevins/SparseGF2`
- Launch profile: `production`, eight BelowNormal workers
- Launch date: 2026-08-10 (America/New_York)

The production run was launched from the long-lived research checkout, whose
Git metadata still names the older `v0.1.0` branch.  Before launch, every
Python file under `src/sparsegf2` and the fingerprinted campaign inputs were
verified against the clean public worktree.  Recomputing the source
fingerprint from published commit `eef62e2` produced the exact value stored in
the run manifest above.

Until this run finishes, do not modify its fingerprinted inputs in the
research checkout: `src/sparsegf2/**/*.py`, `campaign.py`,
`inputs/refinement_centers.csv`, or `single_ref/engine.py`.  Analysis code,
documentation, figures, and ignored outputs are outside the trajectory
fingerprint and can be developed while the simulation continues.
