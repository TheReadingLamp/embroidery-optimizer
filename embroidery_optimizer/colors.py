"""Thread-colour reduction via k-means clustering, implemented in numpy.

Determinism: centroids are initialised with a fixed-seed k-means++ scheme,
so identical input always yields identical output.  Quality is measured as
the usage-weighted quantization error: for every thread colour, its stitch
count times its Euclidean RGB distance to the nearest reduced-palette
colour, summed over all colours.  Lower is better.
"""

import numpy as np

__all__ = ["kmeans", "assign_to_centroids", "quantization_error", "top_k_palette"]


def kmeans(points, k, weights=None, seed=42, max_iter=100):
    """Deterministic k-means.  Returns (centroids, assignments).

    points: (n, d) array.  weights: optional per-point usage weights used
    both for seeding (k-means++ distribution) and centroid updates.
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if k >= n:
        return points.copy(), np.arange(n)
    rng = np.random.default_rng(seed)

    # Fixed-seed k-means++ initialisation.
    centroids = [points[int(rng.integers(n))]]
    d2 = ((points - centroids[0]) ** 2).sum(axis=1)
    while len(centroids) < k:
        probs = d2 * weights
        total = probs.sum()
        probs = np.ones(n) / n if total <= 0.0 else probs / total
        idx = int(rng.choice(n, p=probs))
        centroids.append(points[idx])
        d2 = np.minimum(d2, ((points - centroids[-1]) ** 2).sum(axis=1))
    centroids = np.array(centroids)

    assign = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_assign = dists.argmin(axis=1)
        new_centroids = centroids.copy()
        for c in range(k):
            mask = new_assign == c
            if mask.any():
                w = weights[mask]
                new_centroids[c] = (points[mask] * w[:, None]).sum(axis=0) / w.sum()
        if np.array_equal(new_assign, assign) and np.allclose(new_centroids, centroids):
            assign, centroids = new_assign, new_centroids
            break
        assign, centroids = new_assign, new_centroids
    assign = assign_to_centroids(points, centroids)
    return centroids, assign


def assign_to_centroids(points, centroids):
    """Nearest-centroid index for every point."""
    dists = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return dists.argmin(axis=1)


def quantization_error(points, centroids, weights=None):
    """Usage-weighted total colour error.

    Sum over colours of stitch_count * squared RGB distance to the nearest
    centroid.  This is exactly the objective k-means minimises, so k-means
    can be compared directly against any fixed k-colour palette (e.g. the
    top-k most-used colours).  Lower is better.
    """
    points = np.asarray(points, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    weights = np.ones(len(points)) if weights is None else np.asarray(weights, dtype=float)
    diff = points[:, None, :] - centroids[None, :, :]
    dists2 = (diff ** 2).sum(axis=2).min(axis=1)
    return float((dists2 * weights).sum())


def top_k_palette(points, k, weights=None):
    """Naive baseline: keep the k most-used colours, discard the rest."""
    weights = np.ones(len(points)) if weights is None else np.asarray(weights, dtype=float)
    return points[np.argsort(-weights)[:k]]
