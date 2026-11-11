"""Command-line interface: generate / optimize-colors / optimize-path / report.

Run as:  python -m embroidery_optimizer <command> [options]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

from . import __version__, colors, generator, io, paths

# Assumptions for the machine-time estimate (see README).
TRAVEL_SPEED_MM_S = 250.0     # jump/travel moves
SECONDS_PER_THREAD_CHANGE = 8.0


def cmd_generate(args):
    os.makedirs(args.out, exist_ok=True)
    for i, name in enumerate(generator.DESIGNS[:args.count]):
        design = generator.build_design(name, args.seed + i)
        path = os.path.join(args.out, f"{name}.json")
        io.save_design(design, path)
        print(f"wrote {path}: {design['stitch_count']} stitches, "
              f"{design['color_count']} thread colours")


def _reduce_colors(design, k, seed):
    rgb = io.color_rgb(design)
    weights = io.color_weights(design)
    centroids, assign = colors.kmeans(rgb, k, weights=weights, seed=seed)
    err = colors.quantization_error(rgb, centroids, weights)
    topk = colors.top_k_palette(rgb, k, weights)
    base_err = colors.quantization_error(rgb, topk, weights)
    return centroids, err, base_err


def cmd_optimize_colors(args):
    design = io.load_design(args.design)
    centroids, err, base_err = _reduce_colors(design, args.k, args.seed)
    reduced = io.remap_colors(design, centroids)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    io.save_design(reduced, args.out)
    impr = 100.0 * (base_err - err) / base_err if base_err > 0 else 0.0
    print(f"{design['name']}: {design['color_count']} -> {args.k} colours")
    print(f"  quantization error : {err:.1f} colour-distance units")
    print(f"  top-{args.k}-by-usage baseline error : {base_err:.1f}")
    print(f"  k-means improvement: {impr:.1f}% lower error than baseline")
    print(f"  thread changes: {len(design['color_blocks'])} -> "
          f"{len(reduced['color_blocks'])}")
    print(f"  wrote {args.out}")


def cmd_optimize_path(args):
    design = io.load_design(args.design)
    opt_order, nn_order = paths.optimize_path(design)
    m = paths.path_metrics(design, opt_order, nn_order)
    saved_orig = 100.0 * (m["original_mm"] - m["optimized_mm"]) / m["original_mm"]
    saved_nn = 100.0 * (m["nearest_neighbour_mm"] - m["optimized_mm"]) / m["nearest_neighbour_mm"]
    reordered = paths.reorder_stitches(design, opt_order)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    io.save_design(reordered, args.out)
    print(f"{design['name']}: travel within colour blocks "
          f"{m['original_mm']:.1f} mm -> {m['optimized_mm']:.1f} mm")
    print(f"  nearest-neighbour baseline: {m['nearest_neighbour_mm']:.1f} mm "
          f"(2-opt improves it by {saved_nn:.1f}%)")
    print(f"  saved {saved_orig:.1f}% of as-digitised travel "
          f"({m['original_mm'] - m['optimized_mm']:.1f} mm)")
    print(f"  total incl. jump stitches: {m['total_original_mm']:.1f} mm -> "
          f"{m['total_optimized_mm']:.1f} mm")
    print(f"  thread changes unchanged: {len(design['color_blocks'])}")
    print(f"  wrote {args.out}")


def _row(design, k, seed):
    """Run both optimisations on one design; return a metrics dict."""
    rgb = io.color_rgb(design)
    weights = io.color_weights(design)
    centroids, _, base_err = _reduce_colors(design, k, seed)
    err = colors.quantization_error(rgb, centroids, weights)
    reduced = io.remap_colors(design, centroids)

    opt_order, nn_order = paths.optimize_path(design)
    m = paths.path_metrics(design, opt_order, nn_order)
    orig_changes = len(design["color_blocks"])
    new_changes = len(reduced["color_blocks"])
    travel_saved_mm = m["original_mm"] - m["optimized_mm"]
    err_impr = 100.0 * (base_err - err) / base_err if base_err > 0 else 0.0
    chg_impr = (100.0 * (orig_changes - new_changes) / orig_changes
                if orig_changes > 0 else 0.0)
    return {
        "design": design["name"],
        "stitches": design["stitch_count"],
        "colors_before": design["color_count"],
        "colors_after": k,
        "color_error_kmeans": round(err, 1),
        "color_error_topk_baseline": round(base_err, 1),
        "color_error_improvement_pct": round(err_impr, 1),
        "thread_changes_before": orig_changes,
        "thread_changes_after": new_changes,
        "thread_changes_saved_pct": round(chg_impr, 1),
        "path_mm_original": round(m["original_mm"], 1),
        "path_mm_nn_baseline": round(m["nearest_neighbour_mm"], 1),
        "path_mm_optimized": round(m["optimized_mm"], 1),
        "path_saved_pct": round(100.0 * travel_saved_mm / m["original_mm"], 1),
        "path_saved_vs_nn_pct": round(
            100.0 * (m["nearest_neighbour_mm"] - m["optimized_mm"])
            / m["nearest_neighbour_mm"], 1),
        "path_total_original_mm": round(m["total_original_mm"], 1),
        "path_total_optimized_mm": round(m["total_optimized_mm"], 1),
        "est_time_saved_s": round(
            travel_saved_mm / TRAVEL_SPEED_MM_S
            + (orig_changes - new_changes) * SECONDS_PER_THREAD_CHANGE, 1),
    }


def cmd_report(args):
    files = []
    for pat in args.designs:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        print(f"no design files matched {args.designs}", file=sys.stderr)
        return 1
    rows = [_row(io.load_design(f), args.k, args.seed) for f in files]
    total = {
        "design": "TOTAL",
        "stitches": sum(r["stitches"] for r in rows),
        "colors_before": "-", "colors_after": args.k,
        "color_error_kmeans": round(sum(r["color_error_kmeans"] for r in rows), 1),
        "color_error_topk_baseline": round(sum(r["color_error_topk_baseline"] for r in rows), 1),
        "color_error_improvement_pct": "-",
        "thread_changes_before": sum(r["thread_changes_before"] for r in rows),
        "thread_changes_after": sum(r["thread_changes_after"] for r in rows),
        "thread_changes_saved_pct": "-",
        "path_mm_original": round(sum(r["path_mm_original"] for r in rows), 1),
        "path_mm_nn_baseline": round(sum(r["path_mm_nn_baseline"] for r in rows), 1),
        "path_mm_optimized": round(sum(r["path_mm_optimized"] for r in rows), 1),
        "path_saved_pct": round(100.0 * (sum(r["path_mm_original"] for r in rows)
                                         - sum(r["path_mm_optimized"] for r in rows))
                                / sum(r["path_mm_original"] for r in rows), 1),
        "path_saved_vs_nn_pct": round(
            100.0 * (sum(r["path_mm_nn_baseline"] for r in rows)
                     - sum(r["path_mm_optimized"] for r in rows))
            / sum(r["path_mm_nn_baseline"] for r in rows), 1),
        "est_time_saved_s": round(sum(r["est_time_saved_s"] for r in rows), 1),
    }
    if rows:
        w_err = sum(r["color_error_topk_baseline"] for r in rows)
        if w_err > 0:
            total["color_error_improvement_pct"] = round(
                100.0 * (w_err - total["color_error_kmeans"]) / w_err, 1)
        if total["thread_changes_before"] > 0:
            total["thread_changes_saved_pct"] = round(
                100.0 * (total["thread_changes_before"] - total["thread_changes_after"])
                / total["thread_changes_before"], 1)

    header = (f"{'design':<14}{'stitches':>10}{'cols':>7}{'k':>4}{'err':>9}{'topk':>9}"
              f"{'err+%':>7}{'chg':>6}{'chg+%':>7}{'path0':>10}{'pathnn':>9}{'path1':>9}"
              f"{'p0+%':>7}{'pnn+%':>7}{'secs':>7}")
    print("embroidery-optimizer report  (k=%d, seed=%d)" % (args.k, args.seed))
    print(header)
    print("-" * len(header))
    for r in rows + [total]:
        print(f"{r['design']:<14}{r['stitches']:>10}{str(r['colors_before']):>7}"
              f"{r['colors_after']:>4}{r['color_error_kmeans']:>9}{r['color_error_topk_baseline']:>9}"
              f"{str(r['color_error_improvement_pct']):>7}{r['thread_changes_before']:>6}"
              f"{str(r['thread_changes_saved_pct']):>7}{r['path_mm_original']:>10}"
              f"{r['path_mm_nn_baseline']:>9}{r['path_mm_optimized']:>9}"
              f"{str(r['path_saved_pct']):>7}{str(r['path_saved_vs_nn_pct']):>7}"
              f"{r['est_time_saved_s']:>7}")
    print()
    print("cols: thread colours before/after;  err/topk: quantization error (k-means vs")
    print("top-k-by-usage baseline);  err+%: error reduction;  chg/chg+%: thread changes")
    print("before/after and % saved;  path0/pathnn/path1: travel mm WITHIN colour blocks")
    print("as-digitised / nearest-neighbour baseline / optimised;  p0+%/pnn+%: % saved")
    print("vs each;  secs: est. machine time saved (8 s/thread change, 250 mm/s travel).")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"k": args.k, "seed": args.seed, "rows": rows, "total": total},
                      f, indent=2)
            f.write("\n")
        print(f"wrote {args.out}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="embroidery-optimizer",
        description="AI optimisation for digitised embroidery designs "
                    "(colour reduction + stitch-path optimisation).")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate deterministic sample designs")
    g.add_argument("--out", default="designs", help="output directory (default: designs)")
    g.add_argument("--count", type=int, default=3, choices=range(1, 4))
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("optimize-colors", help="k-means thread-colour reduction")
    c.add_argument("design", help="design JSON file")
    c.add_argument("--k", type=int, default=5)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--out", required=True, help="output design JSON file")
    c.set_defaults(func=cmd_optimize_colors)

    t = sub.add_parser("optimize-path", help="NN + 2-opt stitch-path optimisation")
    t.add_argument("design", help="design JSON file")
    t.add_argument("--out", required=True, help="output design JSON file")
    t.set_defaults(func=cmd_optimize_path)

    r = sub.add_parser("report", help="run both optimisations and print a results table")
    r.add_argument("--designs", nargs="+", default=["designs/*.json"],
                   help="design JSON files or glob patterns")
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--out", default=None, help="optional JSON report output")
    r.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
