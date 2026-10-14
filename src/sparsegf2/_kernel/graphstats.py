"""Breadth-first search over a graph in CSR form, for graphs.stats."""

import numpy as np
from numba import njit


@njit(cache=True)
def path_length_sum(indptr, indices, n):
    """(sum of shortest-path lengths, number of reachable ordered pairs) over all sources."""
    dist = np.empty(n, np.int64)
    queue = np.empty(n, np.int64)
    total = 0
    pairs = 0
    for s in range(n):
        dist[:] = -1
        dist[s] = 0
        head = 0
        tail = 1
        queue[0] = s
        while head < tail:
            u = queue[head]
            head += 1
            for j in range(indptr[u], indptr[u + 1]):
                v = indices[j]
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    total += dist[v]
                    pairs += 1
                    queue[tail] = v
                    tail += 1
    return total, pairs
