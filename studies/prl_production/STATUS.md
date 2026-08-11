# PRL production status

- **Production state:** active locally; live counters are intentionally not versioned
- **Run ID:** `f76def10804a67a6`
- **Simulator review:** complete and published
- **Smoke validation:** complete

- Production grid: 37,400 points and 18,700,000 trajectories.
- Production geometry: Watts-Strogatz `C(n,2)`, exactly `2n` edges and
  mean degree 4.
- Production depth: `T_max=8n`, with `S(R)` evaluated after every measured
  layer.
- Smoke run: 18 / 18 points and 72 / 72 trajectories complete.
- Smoke events: 48 observed purification times and 24 correctly censored
  `p=0` trajectories.
- Resume replay: byte-identical across all 24 NPZ files; aggregate SHA-256
  `6fec0dae233cfb51ed73d1bf98f16a445a220fe46a1296ab3fd21c3f917dfa20`.

The local monitor replaces the research checkout's copy of this document
atomically with live progress and ETA.  See [`PROVENANCE.md`](PROVENANCE.md)
for the immutable source fingerprint and published commit used by the active
run; generated data, live manifests, and logs remain local.
