"""Deterministic sample-design generator (seeded RNG, numpy only).

Each design is built from coloured strokes (polylines sampled from
parametric curves).  Strokes are emitted in a deliberately imperfect
"creation order" -- the order a digitising artist might draw them in --
so that the stitch-path optimiser has real travel distance to reclaim.
Stitches are grouped into colour blocks, like a machine-ready file.

Designs share a base seed (42); design i uses seed 42 + i.
"""

import numpy as np

__all__ = ["DESIGNS", "build_design"]

DESIGNS = ("rose_bloom", "monogram_a", "alpine_scene")

STITCH_LEN = 2.2  # mm between consecutive stitches


def _jitter(rng, pts, amount=0.35):
    return pts + rng.uniform(-amount, amount, size=pts.shape)


def _line(p0, p1, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return np.asarray(p0) * (1 - t) + np.asarray(p1) * t


def _circle(cx, cy, r, n, a0=0.0, a1=2.0 * np.pi):
    t = np.linspace(a0, a1, n)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def _bezier(p0, p1, p2, p3, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    mt = 1.0 - t
    return (mt ** 3 * p0 + 3 * mt ** 2 * t * p1 + 3 * mt * t ** 2 * p2 + t ** 3 * p3)


def _petal(cx, cy, angle, length, width, n=13):
    """Closed almond-shaped petal outline."""
    a = angle
    base = np.array([cx, cy])
    tip = np.array([cx + length * np.cos(a), cy + length * np.sin(a)])
    perp = np.array([-np.sin(a), np.cos(a)])
    half1 = _bezier(base, base + perp * width, tip - perp * width * 0.4, tip, n)
    half2 = _bezier(base, base - perp * width, tip + perp * width * 0.4, tip, n)
    return np.vstack([half1[:-1], half2[::-1]])


def _sample(points, stitch_len):
    """Resample a polyline at ~stitch_len mm spacing."""
    pts = np.asarray(points, dtype=float)
    seg = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    total = seg.sum()
    n = max(2, int(round(total / stitch_len)) + 1)
    if n == len(pts):
        return pts
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return np.column_stack([np.interp(np.linspace(0.0, total, n), cum, pts[:, 0]),
                            np.interp(np.linspace(0.0, total, n), cum, pts[:, 1])])


def _assemble(name, seed, colors, strokes):
    """Group sampled strokes into colour blocks and build the design dict."""
    rng = np.random.default_rng(seed)
    # sample + jitter every stroke
    sampled = []
    for color, pts in strokes:
        s = _jitter(rng, _sample(pts, STITCH_LEN))
        if len(s) >= 2:
            sampled.append((color, s))
    # order colours by first appearance; strokes of one colour merge into
    # one contiguous block, keeping their (deliberately imperfect) creation
    # order inside the block
    colour_order, seen, by_colour = [], set(), {}
    for color, pts in sampled:
        if color not in seen:
            seen.add(color)
            colour_order.append(color)
        by_colour.setdefault(color, []).append(pts)
    blocks, stitches = [], []
    for color in colour_order:
        blocks.append({"color": color, "start": len(stitches), "end": 0})
        for pts in by_colour[color]:
            for x, y in pts:
                stitches.append({"i": len(stitches), "x": round(float(x), 2),
                                 "y": round(float(y), 2), "color": color})
        blocks[-1]["end"] = len(stitches)
    counts = [0] * len(colors)
    for s in stitches:
        counts[s["color"]] += 1
    xs = [s["x"] for s in stitches]
    ys = [s["y"] for s in stitches]
    return {
        "name": name,
        "seed": seed,
        "generator": "embroidery-optimizer/1.0",
        "stitch_length_mm": STITCH_LEN,
        "width_mm": max(1, int(round(max(xs) - min(xs)))),
        "height_mm": max(1, int(round(max(ys) - min(ys)))),
        "stitch_count": len(stitches),
        "color_count": len(colors),
        "thread_colors": [
            {"index": i, "name": cname, "rgb": [int(v) for v in rgb],
             "stitch_count": counts[i]}
            for i, (cname, rgb) in enumerate(colors)
        ],
        "color_blocks": blocks,
        "stitches": stitches,
    }


# ---------------------------------------------------------------- designs


def _rose(seed):
    cx, cy = 60.0, 58.0
    colors = [
        ("Maize", (240, 205, 70)), ("Crimson", (186, 32, 55)), ("Rose", (232, 106, 118)),
        ("Salmon", (242, 150, 130)), ("Dark Red", (128, 20, 38)), ("Blush", (245, 180, 190)),
        ("Ruby", (200, 60, 80)), ("Leaf Green", (60, 140, 60)), ("Dark Green", (35, 90, 45)),
        ("Stem Green", (90, 160, 70)), ("Charcoal", (40, 40, 45)), ("Gold", (212, 175, 55)),
    ]
    strokes = []
    # centre spiral (colour 0)
    t = np.linspace(0.0, 4.0 * np.pi, 50)
    r = 1.5 + t / (4.0 * np.pi) * 13.0
    strokes.append((0, np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])))
    # inner ring: 6 petals, colours 1-3, drawn in scrambled creation order
    for k, a in enumerate([np.pi * 0.35, np.pi * 1.25, np.pi * 2.35, np.pi * 0.85,
                           np.pi * 1.85, np.pi * 0.05]):
        strokes.append((1 + (k % 3), _petal(cx, cy, a, 27.0, 10.0)))
    # outer ring: 8 petals, colours 4-6, deliberately scattered so that each
    # colour block jumps back and forth across the bloom
    for k, a in enumerate([np.pi * 1.15, np.pi * 2.85, np.pi * 0.55, np.pi * 1.75,
                           np.pi * 2.45, np.pi * 0.15, np.pi * 0.95, np.pi * 1.55]):
        strokes.append((4 + (k % 3), _petal(cx, cy, a, 41.0, 14.0)))
    # two leaves below the bloom (colours 7, 8)
    strokes.append((7, _petal(cx, cy + 8.0, -0.85 * np.pi / 2.0 - 0.25, 46.0, 13.0)))
    strokes.append((8, _petal(cx, cy + 8.0, -0.5 * np.pi / 2.0 + 0.28, 44.0, 12.0)))
    # stem (9)
    strokes.append((9, _line((cx, cy + 10.0), (cx, cy + 46.0), 14)))
    # border circle (10) and three gold accent arcs (11)
    strokes.append((10, _circle(cx, cy, 55.0, int(2 * np.pi * 55.0 / STITCH_LEN))))
    for a in (0.4, 2.3, 4.3):
        strokes.append((11, _circle(cx, cy, 49.0, 10, a, a + 0.8)))
    return _assemble("rose_bloom", seed, colors, strokes)


def _monogram(seed):
    colors = [
        ("Navy", (25, 35, 85)), ("Royal Blue", (45, 85, 200)), ("Sky Blue", (135, 190, 240)),
        ("Gold", (212, 175, 55)), ("White", (245, 245, 245)), ("Charcoal", (40, 40, 45)),
        ("Teal", (30, 140, 140)), ("Ruby", (190, 40, 60)), ("Olive", (130, 130, 50)),
        ("Forest", (40, 100, 50)), ("Silver", (190, 195, 205)), ("Denim", (60, 80, 150)),
    ]
    # letter A geometry: left/right diagonals and a crossbar
    left = _line((32.0, 100.0), (60.0, 22.0), 24)
    right = _line((88.0, 100.0), (60.0, 22.0), 24)
    bar = _line((41.0, 64.0), (79.0, 64.0), 14)

    def offsets(line, widths):
        d = line[-1] - line[0]
        p = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        return [line + p * w for w in widths]

    strokes = []
    # Creation order deliberately non-optimal: crossbar first, then right,
    # then left diagonal; each stroke is a 3-pass satin fill.
    off_b = offsets(bar, (-2.5, 0.0, 2.5))
    strokes.append((11, off_b[0]))  # Denim outer
    strokes.append((0, off_b[1]))   # Navy middle
    strokes.append((1, off_b[2]))   # Royal Blue inner
    # right diagonal: Navy / Royal Blue / Sky Blue, plus a gold trim line
    off_r = offsets(right, (-3.0, 0.0, 3.0))
    strokes.append((0, off_r[0]))
    strokes.append((1, off_r[1]))
    strokes.append((2, off_r[2]))
    strokes.append((3, offsets(right, (-5.5,))[0]))
    # left diagonal: Navy / Royal Blue / Sky Blue, plus a white highlight
    off_l = offsets(left, (-3.0, 0.0, 3.0))
    strokes.append((0, off_l[0]))
    strokes.append((1, off_l[1]))
    strokes.append((2, off_l[2]))
    strokes.append((4, offsets(left, (5.5,))[0]))
    # border circle + teal corner arcs + laurel branches + gems
    strokes.append((5, _circle(60.0, 60.0, 55.0, int(2 * np.pi * 55.0 / STITCH_LEN))))
    for a in (2.0, 4.6):
        strokes.append((6, _circle(60.0, 60.0, 46.0, 12, a, a + 1.1)))
    strokes.append((8, _bezier((24.0, 100.0), (26.0, 72.0), (30.0, 60.0), (34.0, 50.0), 18)))
    strokes.append((9, _bezier((96.0, 100.0), (94.0, 72.0), (90.0, 60.0), (86.0, 50.0), 18)))
    strokes.append((7, _circle(60.0, 92.0, 3.2, 10)))
    strokes.append((10, _circle(28.0, 32.0, 2.0, 8)))
    strokes.append((10, _circle(92.0, 32.0, 2.0, 8)))
    return _assemble("monogram_a", seed, colors, strokes)


def _alpine(seed):
    colors = [
        ("Pale Sky", (200, 225, 245)), ("Sky", (120, 175, 220)), ("Deep Sky", (60, 110, 175)),
        ("Sun Yellow", (245, 200, 60)), ("Orange", (235, 140, 40)), ("Pale Gold", (225, 195, 110)),
        ("Grey", (120, 120, 130)), ("Brown", (120, 85, 60)), ("Slate", (90, 100, 115)),
        ("Snow White", (250, 250, 250)), ("Grass", (90, 150, 70)), ("Pine", (40, 95, 55)),
    ]
    strokes = []
    # sky bands (0,1,2), top to bottom
    for k, y in enumerate((28.0, 46.0, 64.0)):
        strokes.append((k, _line((6.0, y), (134.0, y - 6.0 * k), 40)))
    # sun (3) with inner ring (4) and eight rays (5)
    strokes.append((3, _circle(112.0, 26.0, 12.0, 26)))
    strokes.append((4, _circle(112.0, 26.0, 6.5, 16)))
    for k in range(8):
        a = k * np.pi / 4.0
        strokes.append((5, _line((112.0 + 15.0 * np.cos(a), 26.0 + 15.0 * np.sin(a)),
                                 (112.0 + 19.5 * np.cos(a), 26.0 + 19.5 * np.sin(a)), 4)))
    # three mountain ridges (6,7,8) as zigzag polylines, creation order shuffled
    m0 = np.array([[6.0, 80.0], [30.0, 46.0], [48.0, 62.0], [66.0, 38.0],
                   [84.0, 60.0], [102.0, 48.0], [118.0, 66.0], [134.0, 80.0]])
    m1 = np.array([[6.0, 92.0], [26.0, 70.0], [44.0, 82.0], [64.0, 58.0],
                   [82.0, 78.0], [100.0, 66.0], [116.0, 82.0], [134.0, 92.0]])
    m2 = np.array([[6.0, 100.0], [34.0, 86.0], [58.0, 95.0], [80.0, 80.0],
                   [104.0, 94.0], [134.0, 100.0]])
    for color, m in ((8, m2), (6, m0), (7, m1)):
        strokes.append((color, m))
    # snow caps on the peaks (9)
    for px, py in ((66.0, 38.0), (102.0, 48.0), (64.0, 58.0)):
        strokes.append((9, _line((px - 6.0, py + 5.0), (px + 6.0, py + 5.0), 6)))
    # foreground grass (10, 11) and three pines (11); the digitizer drew all
    # trunks first, then all left branches, then all right branches -- a
    # realistic but wasteful order the path optimiser can fix
    for y in (98.0, 106.0):
        strokes.append((10, _line((6.0, y), (134.0, y), 40)))
    for px in (22.0, 58.0, 92.0):
        strokes.append((11, _line((px, 98.0), (px, 78.0), 8)))
    for px in (22.0, 58.0, 92.0):
        strokes.append((11, _line((px, 88.0), (px - 7.0, 81.0), 5)))
    for px in (22.0, 58.0, 92.0):
        strokes.append((11, _line((px, 88.0), (px + 7.0, 81.0), 5)))
    return _assemble("alpine_scene", seed, colors, strokes)


_BUILDERS = {"rose_bloom": _rose, "monogram_a": _monogram, "alpine_scene": _alpine}


def build_design(name, seed=42):
    """Build a deterministic design dict for `name` (one of DESIGNS)."""
    if name not in _BUILDERS:
        raise KeyError(f"unknown design {name!r}; choose from {list(_BUILDERS)}")
    return _BUILDERS[name](seed)
