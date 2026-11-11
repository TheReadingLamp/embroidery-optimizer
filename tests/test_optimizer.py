"""Plain-assert test script:  python tests/test_optimizer.py

Covers the deterministic generator, the numpy k-means colour reduction,
the NN + 2-opt path optimisation, and CLI smoke tests.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from embroidery_optimizer import colors, generator, io, paths
from embroidery_optimizer.geometry import (nearest_neighbour, pairwise_distances,
                                           path_length, two_opt)

PASS = 0


def check(name, fn):
    global PASS
    try:
        fn()
    except AssertionError as e:
        print(f"FAIL {name}: {e}")
        raise
    PASS += 1
    print(f"ok   {name}")


# ---------------------------------------------------------------- generator

def test_generator_deterministic():
    a = generator.build_design("rose_bloom", 42)
    b = generator.build_design("rose_bloom", 42)
    assert json.dumps(a) == json.dumps(b), "same seed must give identical design"
    c = generator.build_design("rose_bloom", 43)
    assert json.dumps(a) != json.dumps(c), "different seed should differ"


def test_design_structure():
    for name in generator.DESIGNS:
        d = generator.build_design(name, 42)
        n = d["stitch_count"]
        assert n >= 200, f"{name}: too few stitches ({n})"
        assert d["color_count"] == 12, f"{name}: expected 12 colours"
        colors_seen = set()
        for s in d["stitches"]:
            assert 0 <= s["color"] < 12, f"{name}: bad colour index"
            assert 0 <= s["x"] < 160 and 0 <= s["y"] < 140, f"{name}: out of canvas"
            colors_seen.add(s["color"])
        assert len(colors_seen) >= 8, f"{name}: colours barely used"
        # blocks tile the stitch list exactly, in order, colour-consistent
        pos = 0
        for b in d["color_blocks"]:
            assert b["start"] == pos and b["end"] > b["start"]
            for s in d["stitches"][b["start"]:b["end"]]:
                assert s["color"] == b["color"]
            pos = b["end"]
        assert pos == n
        # stitch counts are consistent
        counts = [0] * 12
        for s in d["stitches"]:
            counts[s["color"]] += 1
        for c in d["thread_colors"]:
            assert c["stitch_count"] == counts[c["index"]]


# ---------------------------------------------------------------- colours

def test_kmeans_deterministic():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 255, (20, 3))
    w = rng.uniform(1, 100, 20)
    c1, a1 = colors.kmeans(pts, 4, weights=w, seed=42)
    c2, a2 = colors.kmeans(pts, 4, weights=w, seed=42)
    assert np.allclose(c1, c2) and np.array_equal(a1, a2)


def test_kmeans_recovers_clusters():
    rng = np.random.default_rng(7)
    true = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype=float)
    pts = np.vstack([true[i] + rng.normal(0, 2, (25, 2)) for i in range(4)])
    centroids, assign = colors.kmeans(pts, 4, seed=42)
    diff = centroids[:, None, :] - true[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    assert d.min(axis=1).max() < 5.0, "centroid far from true cluster centre"
    mapped = {int(np.argmin(d[i])) for i in range(4)}
    assert mapped == {0, 1, 2, 3}, "centroids should map onto all four centres"
    # each true cluster lands in a single, distinct assignment
    got = set()
    for i in range(4):
        got.add(int(assign[i * 25:(i + 1) * 25].min()))
        assert assign[i * 25:(i + 1) * 25].min() == assign[i * 25:(i + 1) * 25].max()
    assert len(got) == 4, "assignments should separate the clusters"


def test_quantization_error_known():
    pts = np.array([[0.0, 0.0], [2.0, 0.0]])
    assert colors.quantization_error(pts, np.array([[1.0, 0.0]]), [1, 1]) == 2.0
    assert colors.quantization_error(pts, np.array([[0.0, 0.0]]), [1, 1]) == 4.0
    assert colors.quantization_error(pts, pts, [1, 1]) == 0.0


def test_kmeans_beats_topk_baseline():
    for name in generator.DESIGNS:
        d = generator.build_design(name, 42)
        rgb = io.color_rgb(d)
        w = io.color_weights(d)
        centroids, _ = colors.kmeans(rgb, 5, weights=w, seed=42)
        err = colors.quantization_error(rgb, centroids, w)
        base = colors.quantization_error(rgb, colors.top_k_palette(rgb, 5, w), w)
        assert err < base, f"{name}: k-means ({err:.1f}) should beat top-k ({base:.1f})"


# ---------------------------------------------------------------- paths

def test_path_length_known():
    pts = np.array([[0, 0], [3, 0], [3, 4]])
    d = pairwise_distances(pts)
    assert path_length(d, [0, 1, 2]) == 7.0
    assert path_length(d, [0, 2, 1]) == 9.0
    assert path_length(d, [0]) == 0.0


def test_nn_and_two_opt_square():
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    d = pairwise_distances(pts)
    nn = nearest_neighbour(d, 0)
    assert path_length(d, nn) == 30.0, "NN on a square is 3 sides"
    opt = two_opt(d, nn)
    assert path_length(d, opt) <= path_length(d, nn)
    assert path_length(d, opt) == 30.0, "square NN order is already 2-opt optimal"


def test_two_opt_strictly_improves():
    # on a fixed-seed random instance the greedy NN order is essentially
    # never 2-opt-optimal, so 2-opt must find a strictly shorter path
    rng = np.random.default_rng(123)
    pts = rng.uniform(0, 100, (30, 2))
    d = pairwise_distances(pts)
    nn = nearest_neighbour(d, 0)
    opt = two_opt(d, nn)
    assert path_length(d, opt) < path_length(d, nn) - 1e-9, \
        "2-opt must shorten the NN order"
    assert sorted(opt) == list(range(30)), "2-opt must keep a permutation"


def test_two_opt_never_worsens():
    rng = np.random.default_rng(11)
    for _ in range(5):
        pts = rng.uniform(0, 100, (25, 2))
        d = pairwise_distances(pts)
        order = list(rng.permutation(25))
        improved = two_opt(d, order)
        assert path_length(d, improved) <= path_length(d, order) + 1e-9


def test_optimize_path_preserves_blocks_and_shortens():
    for name in generator.DESIGNS:
        d = generator.build_design(name, 42)
        opt, nn = paths.optimize_path(d)
        n = d["stitch_count"]
        assert sorted(opt) == list(range(n)), f"{name}: not a permutation"
        # colour block structure (run-length sequence of colours) unchanged
        def runs(order):
            out = []
            for i in order:
                c = d["stitches"][i]["color"]
                if not out or out[-1][0] != c:
                    out.append([c, 0])
                out[-1][1] += 1
            return out
        assert runs(opt) == runs(nn) == runs(list(range(n))), \
            f"{name}: colour blocks changed"
        m = paths.path_metrics(d, opt, nn)
        assert m["optimized_mm"] <= m["nearest_neighbour_mm"] + 1e-9
        assert m["optimized_mm"] <= m["original_mm"] + 1e-9, \
            (f"{name}: optimised path worse than as-digitised "
             f"({m['optimized_mm']} vs {m['original_mm']})")


def test_remap_colors():
    d = generator.build_design("rose_bloom", 42)
    rgb = io.color_rgb(d)
    w = io.color_weights(d)
    centroids, _ = colors.kmeans(rgb, 5, weights=w, seed=42)
    reduced = io.remap_colors(d, centroids)
    assert reduced["color_count"] == 5
    assert len(reduced["thread_colors"]) == 5
    assert reduced["stitch_count"] == d["stitch_count"]
    for s in reduced["stitches"]:
        assert 0 <= s["color"] < 5
    assert sum(c["stitch_count"] for c in reduced["thread_colors"]) == d["stitch_count"]
    assert len(reduced["color_blocks"]) < len(d["color_blocks"])


# ---------------------------------------------------------------- CLI

def test_cli_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, PYTHONPATH=ROOT)
        def run(*argv):
            return subprocess.run([sys.executable, "-m", "embroidery_optimizer"] + list(argv),
                                  cwd=ROOT, env=env, capture_output=True, text=True)
        r = run("generate", "--out", tmp, "--count", "3", "--seed", "42")
        assert r.returncode == 0, r.stderr
        for name in generator.DESIGNS:
            assert os.path.exists(os.path.join(tmp, f"{name}.json"))
        r = run("optimize-colors", os.path.join(tmp, "rose_bloom.json"),
                "--k", "5", "--out", os.path.join(tmp, "rose_c5.json"))
        assert r.returncode == 0, r.stderr
        assert os.path.exists(os.path.join(tmp, "rose_c5.json"))
        r = run("optimize-path", os.path.join(tmp, "rose_bloom.json"),
                "--out", os.path.join(tmp, "rose_p.json"))
        assert r.returncode == 0, r.stderr
        assert os.path.exists(os.path.join(tmp, "rose_p.json"))
        r = run("report", "--designs", os.path.join(tmp, "*.json"),
                "--k", "5", "--out", os.path.join(tmp, "report.json"))
        assert r.returncode == 0, r.stderr
        assert os.path.exists(os.path.join(tmp, "report.json"))
        rep = json.load(open(os.path.join(tmp, "report.json"), encoding="utf-8"))
        assert rep["total"]["path_saved_pct"] > 0
        assert rep["total"]["color_error_improvement_pct"] > 0


if __name__ == "__main__":
    for fn in [test_generator_deterministic, test_design_structure,
               test_kmeans_deterministic, test_kmeans_recovers_clusters,
               test_quantization_error_known, test_kmeans_beats_topk_baseline,
               test_path_length_known, test_nn_and_two_opt_square,
               test_two_opt_strictly_improves, test_two_opt_never_worsens,
               test_optimize_path_preserves_blocks_and_shortens,
               test_remap_colors, test_cli_smoke]:
        check(fn.__name__, fn)
    print(f"\nall {PASS} tests passed")
