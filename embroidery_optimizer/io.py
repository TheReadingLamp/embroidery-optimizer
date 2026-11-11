"""Design file load/save, metrics and palette remapping."""

import json

import numpy as np

__all__ = ["load_design", "save_design", "stitch_matrix", "color_rgb",
           "color_weights", "remap_colors"]


def load_design(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_design(design, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(design, f, indent=2)
        f.write("\n")


def stitch_matrix(design):
    return np.array([[s["x"], s["y"]] for s in design["stitches"]], dtype=float)


def color_rgb(design):
    return np.array([c["rgb"] for c in design["thread_colors"]], dtype=float)


def color_weights(design):
    return np.array([c["stitch_count"] for c in design["thread_colors"]], dtype=float)


def remap_colors(design, centroids):
    """Return a copy of `design` whose palette is the reduced `centroids`.

    Every original thread colour maps to its nearest centroid, and stitches
    are regrouped by reduced colour, so a k-colour design has at most k
    colour blocks (thread changes).
    """
    from .colors import assign_to_centroids

    centroids = np.asarray(centroids, dtype=float)
    mapping = assign_to_centroids(color_rgb(design), centroids)
    k = len(centroids)

    new = {
        "name": design["name"],
        "seed": design["seed"],
        "generator": design["generator"],
        "stitch_length_mm": design["stitch_length_mm"],
        "width_mm": design["width_mm"],
        "height_mm": design["height_mm"],
        "stitch_count": design["stitch_count"],
        "color_count": k,
        "reduced_from_colors": design["color_count"],
        "thread_colors": [
            {"index": i, "name": f"Blend {i + 1}",
             "rgb": [int(round(v)) for v in centroids[i]],
             "stitch_count": 0}
            for i in range(k)
        ],
        "color_blocks": [],
        "stitches": [],
    }
    counts = [0] * k
    for c in range(k):
        start = len(new["stitches"])
        for s in design["stitches"]:
            if int(mapping[s["color"]]) == c:
                counts[c] += 1
                new["stitches"].append({"i": len(new["stitches"]), "x": s["x"],
                                        "y": s["y"], "color": c})
        if len(new["stitches"]) > start:
            new["color_blocks"].append({"color": c, "start": start,
                                        "end": len(new["stitches"])})
    for i, c in enumerate(counts):
        new["thread_colors"][i]["stitch_count"] = c
    return new
