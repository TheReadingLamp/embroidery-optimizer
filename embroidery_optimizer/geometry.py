"""Geometry helpers for stitch-path optimisation (numpy only).

A stitch path is an ordered sequence of (x, y) points; the machine travels
the straight-line distance between consecutive points.  We model travel cost
as the sum of those segment lengths in millimetres.
"""

import numpy as np

__all__ = ["pairwise_distances", "path_length", "nearest_neighbour", "two_opt"]


def pairwise_distances(points):
    """Euclidean distance matrix of shape (n, n) for points of shape (n, 2)."""
    points = np.asarray(points, dtype=float)
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.maximum((diff ** 2).sum(axis=2), 0.0))


def path_length(distance_matrix, order):
    """Total travel distance following `order` (sequence of point indices)."""
    if len(order) < 2:
        return 0.0
    o = np.asarray(order, dtype=int)
    return float(distance_matrix[o[:-1], o[1:]].sum())


def nearest_neighbour(distance_matrix, start=0):
    """Greedy nearest-neighbour path starting at point `start`.

    Returns a list of point indices.  Deterministic: ties are broken by
    lowest index (numpy argmin).
    """
    d = distance_matrix
    n = d.shape[0]
    unvisited = np.ones(n, dtype=bool)
    order = [int(start)]
    unvisited[start] = False
    current = int(start)
    for _ in range(n - 1):
        dists = np.where(unvisited, d[current], np.inf)
        nxt = int(np.argmin(dists))
        order.append(nxt)
        unvisited[nxt] = False
        current = nxt
    return order


def two_opt(distance_matrix, order, max_passes=None):
    """Best-improvement 2-opt local search for an OPEN path.

    Each pass scans all pairs of edges (a,b), (c,d) with c after b; if
    replacing them with (a,c), (b,d) -- i.e. reversing the segment between
    b and c -- shortens the path, the best such move is applied.  Repeats
    until no improving move exists.  Never increases path length.

    The scan is vectorised per row with numpy, so a pass over n points is
    one O(n) python loop of O(n) numpy operations.
    """
    d = distance_matrix
    o = np.asarray(list(order), dtype=int)
    n = len(o)
    if n < 4:
        return o.tolist()
    limit = max_passes if max_passes is not None else max(8, n)
    for _ in range(limit):
        best_gain = 0.0
        best = None
        for i in range(n - 3):  # i+1 <= n-2, and j ranges i+2..n-2
            a, b = int(o[i]), int(o[i + 1])
            js = np.arange(i + 2, n - 1)
            gain = d[a, o[js]] + d[b, o[js + 1]] - d[a, b] - d[o[js], o[js + 1]]
            k = int(np.argmin(gain))
            g = float(gain[k])
            if g < best_gain - 1e-12:
                best_gain = g
                best = (i, int(js[k]))
        if best is None:
            break
        i, j = best
        o[i + 1:j + 1] = o[i + 1:j + 1][::-1]
    return o.tolist()
