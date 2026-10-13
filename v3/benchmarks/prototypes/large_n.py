import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layouts_v2_vs_dual import make_circuit, new_sim, old_sim
from layouts_column_and_stim import col_sim
n = int(sys.argv[1])
for p in (0.2, 0.3, 0.5):
    circ = make_circuit(n, p, n, seed=3)
    t_old = old_sim(n, circ)[0]
    t_dual = new_sim(n, circ, False)[0]
    t_col = col_sim(n, circ, False)[0]
    print(f"n={n} p={p} depth={n}: old={t_old:.2f}s dual1st={t_dual:.2f}s col1st={t_col:.2f}s", flush=True)
