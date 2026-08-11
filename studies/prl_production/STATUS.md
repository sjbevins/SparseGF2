# PRL production status

- **Production state:** not launched
- **Simulator review:** complete; publication approved and being finalized
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

The production launcher requires an explicit confirmation flag.  After the
reviewed commits are published, the monitor will replace this document
atomically with live progress and ETA.
