# Graph registry status

Status: **seed collection complete and validated**

- Collection: `ws_c2_paper_v1_cf9d71dc40b2ab1c`
- Sizes: `64, 96, 128, 160, 192, 224, 256`
- Beta grid: `50` values (`0` plus `49` log-spaced
  values from `0.005` through `1`)
- Cells: `350`
- Indexed graph draws per cell: `1000`
- Total indexed graph draws: `350,000`
- Graph construction: Watts-Strogatz rewiring of `C(n,2)`, mean degree `4`
- Seed-table SHA-256: `4160c96be89f5517bdf97d54ab8c93e8dce768c8dce32e1cbd7cf4889ad76c75`
- SQLite integrity: `ok`; foreign-key violations:
  `0`
- Representative graph reconstructions checked: `35`

The SQLite catalog is at `studies/prl_production/data/graph_registry/ws_c2_paper_v1_cf9d71dc40b2ab1c/graph_registry.sqlite3`.  Its graph rows are the
canonical collection.  Edge banks, invariants, and circuit results are separate,
versioned additions linked to these immutable graph identities.  Equal seeds or
equal topologies never merge records.  At `beta=0`, all indexed draws correctly
reconstruct the same unrewired `C(n,2)` geometry.
