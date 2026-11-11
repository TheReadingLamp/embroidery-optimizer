"""Stitch-path optimisation: per-colour-block nearest neighbour + 2-opt.

Colour blocks are preserved exactly (a block is still one contiguous run),
so the number of thread changes is unchanged.  Inside each block, stitches
are reordered to minimise travel distance, which is what the machine pays
for in jump stitches and running time.
"""

import numpy as np

from .geometry import nearest_neighbour, pairwise_distances, path_length, two_opt
from .io import stitch_matrix

__all__ = ["optimize_path", "path_metrics"]


def _blockwise_order(xy, blocks, apply_two_opt):
    order = []
    for b in blocks:
        idx = list(range(b["start"], b["end"]))
        m = len(idx)
        if m >= 2:
            d = pairwise_distances(xy[idx])
            start = int(np.argmin(d.sum(axis=1)))  # most central stitch
            nn = nearest_neighbour(d, start)
            if apply_two_opt and m >= 4:
                # 2-opt from both the NN order and the as-digitised order;
                # keep the better result, so the optimised path is never
                # worse than the file as shipped.
                best_order, best_len = None, np.inf
                for cand in (nn, list(range(m))):
                    t = two_opt(d, cand)
                    length = path_length(d, t)
                    if length < best_len - 1e-12:
                        best_order, best_len = t, length
                local = best_order
            else:
                local = nn
            order.extend(idx[i] for i in local)
        else:
            order.extend(idx)
    return order


def _choose_block_orientations(order, distance_matrix, blocks):
    """Greedily flip whole blocks to shorten the jump stitches around them.

    The stitch-in/stitch-out direction of a block is free: reversing the
    whole block keeps its internal travel identical but changes the jump
    distance from the previous block and to the next one.
    """
    d = distance_matrix
    o = list(order)
    for k, b in enumerate(blocks):
        if b["end"] - b["start"] < 2:
            continue
        s, e = b["start"], b["end"] - 1
        prev = o[blocks[k - 1]["end"] - 1] if k > 0 else None
        nxt = o[blocks[k + 1]["start"]] if k + 1 < len(blocks) else None
        cur = (d[prev, o[s]] if prev is not None else 0.0) + \
              (d[o[e], nxt] if nxt is not None else 0.0)
        alt = (d[prev, o[e]] if prev is not None else 0.0) + \
              (d[o[s], nxt] if nxt is not None else 0.0)
        if alt < cur - 1e-12:
            o[s:e + 1] = o[s:e + 1][::-1]
    return o


def optimize_path(design):
    """Return (optimized_order, nn_baseline_order).

    Both are full permutations of stitch indices with identical colour-block
    structure.  The optimised order applies NN + 2-opt inside every block
    (keeping the better of the NN-started and as-digitised-started search)
    and then flips whole blocks where that shortens the jump stitches.
    """
    xy = stitch_matrix(design)
    blocks = design["color_blocks"]
    nn_order = _blockwise_order(xy, blocks, apply_two_opt=False)
    opt_order = _blockwise_order(xy, blocks, apply_two_opt=True)
    opt_order = _choose_block_orientations(opt_order, pairwise_distances(xy), blocks)
    return opt_order, nn_order


def path_metrics(design, opt_order, nn_order):
    """Travel lengths (mm).  Primary metric: travel WITHIN colour blocks
    (the quantity the path optimiser controls).  Totals include the jump
    stitches between blocks."""
    xy = stitch_matrix(design)
    d = pairwise_distances(xy)
    original = list(range(len(xy)))
    blocks = design["color_blocks"]

    def within(o):
        return sum(path_length(d, o[b["start"]:b["end"]]) for b in blocks)

    def jumps(o):
        return sum(d[o[b["end"] - 1], o[blocks[i + 1]["start"]]]
                   for i, b in enumerate(blocks[:-1]))

    w_orig, w_nn, w_opt = within(original), within(nn_order), within(opt_order)
    j_orig, j_opt = jumps(original), jumps(opt_order)
    return {
        "original_mm": w_orig,
        "nearest_neighbour_mm": w_nn,
        "optimized_mm": w_opt,
        "jumps_original_mm": j_orig,
        "jumps_optimized_mm": j_opt,
        "total_original_mm": w_orig + j_orig,
        "total_optimized_mm": w_opt + j_opt,
    }


def reorder_stitches(design, order):
    """Return a copy of `design` with stitches in `order` (blocks rebuilt)."""
    new = dict(design)
    new["stitches"] = []
    new["color_blocks"] = []
    current = -1
    for i, idx in enumerate(order):
        s = design["stitches"][idx]
        if s["color"] != current:
            new["color_blocks"].append({"color": s["color"], "start": i, "end": 0})
            current = s["color"]
        new["stitches"].append({"i": i, "x": s["x"], "y": s["y"], "color": s["color"]})
        new["color_blocks"][-1]["end"] = i + 1
    return new
