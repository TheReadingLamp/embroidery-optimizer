# Embroidery Design Optimizer

An AI optimisation tool for **digitised embroidery designs**: it reduces the
thread palette with k-means clustering and reorders stitches with a
nearest-neighbour + 2-opt travelling-salesman search, then reports the
measurable savings in colour error, thread changes and travel distance.

Python 3.14 · the only third-party dependency is **numpy**. Everything is
deterministic (fixed seeds), so every run reproduces the numbers below.

---

## What is embroidery digitisation?

An embroidery machine does not read images. It reads a **stitch file** — a
list of needle-down points in millimetres, each assigned to a thread colour,
grouped into **colour blocks** (one contiguous run of stitches per thread).
Converting artwork into that file is *digitising*: tracing shapes into
strokes of stitches, choosing stitch length and density, and choosing the
order in which the machine sews everything.

Two properties of a stitch file dominate machine time and cost:

1. **Thread changes.** Every colour block means the machine stops, trims the
   thread, and a new cone is loaded (or changed by the operator) — roughly
   6–12 seconds each on a production machine, plus idle labour on a
   single-needle home machine. A 12-colour design costs ~11 extra stops
   compared with a 5-colour design.
2. **Travel distance.** The machine moves the needle along the straight-line
   distance between consecutive stitches, at a fixed travel speed. Stitches
   inside a shape are unavoidable, but *jump stitches* — long moves between
   disconnected elements of the same colour — are pure waste created by a
   bad stitch order. The optimiser cannot remove points, but it can pick the
   order that minimises the distance travelled between them.

This project attacks both: **colour reduction** (12 colours → 5, with a
measured colour-quality cost) and **stitch-path optimisation** (NN + 2-opt
inside each colour block, so thread changes stay untouched while travel
shrinks).

## Project layout

```
embroidery-optimizer/
├── embroidery_optimizer/
│   ├── __init__.py      package metadata
│   ├── generator.py     deterministic sample-design generator (seeded RNG)
│   ├── colors.py        numpy k-means, quantization error, top-k baseline
│   ├── geometry.py      distance matrix, path length, NN, 2-opt
│   ├── paths.py         per-block NN + 2-opt optimisation, path metrics
│   ├── io.py            design load/save, palette remapping
│   ├── cli.py           generate / optimize-colors / optimize-path / report
│   └── __main__.py      python -m embroidery_optimizer entry point
├── designs/             the 3 generated sample designs (JSON)
├── optimized/           colour-reduced and path-optimised outputs (JSON)
├── results/             optimization_report.json (machine-readable metrics)
├── tests/test_optimizer.py   plain-assert test script
└── README.md
```

## Install / requirements

- Windows (or any OS) with **Python 3.14**
- `pip install numpy` (tested with numpy 2.5)

## Usage

All commands run from the project root.

```text
python -m embroidery_optimizer generate --out designs --count 3 --seed 42
python -m embroidery_optimizer optimize-colors designs/rose_bloom.json --k 5 --out optimized/rose_bloom_c5.json
python -m embroidery_optimizer optimize-path designs/rose_bloom.json --out optimized/rose_bloom_p.json
python -m embroidery_optimizer report --designs "designs/*.json" --k 5 --out results/optimization_report.json
python tests/test_optimizer.py
```

### Sample output — `generate`

```text
wrote designs\rose_bloom.json: 873 stitches, 12 thread colours
wrote designs\monogram_a.json: 644 stitches, 12 thread colours
wrote designs\alpine_scene.json: 685 stitches, 12 thread colours
```

### Sample output — `optimize-colors designs/rose_bloom.json --k 5`

```text
rose_bloom: 12 -> 5 colours
  quantization error : 1106128.2 colour-distance units
  top-5-by-usage baseline error : 1549339.0
  k-means improvement: 28.6% lower error than baseline
  thread changes: 12 -> 5
  wrote optimized/rose_bloom_c5.json
```

### Sample output — `optimize-path designs/rose_bloom.json`

```text
rose_bloom: travel within colour blocks 1977.3 mm -> 1920.9 mm
  nearest-neighbour baseline: 2125.1 mm (2-opt improves it by 9.6%)
  saved 2.8% of as-digitised travel (56.3 mm)
  total incl. jump stitches: 2098.7 mm -> 2148.3 mm
  thread changes unchanged: 12
  wrote optimized/rose_bloom_p.json
```

### Design file format (excerpt from `designs/rose_bloom.json`)

```json
{
  "name": "rose_bloom",
  "seed": 42,
  "stitch_length_mm": 2.2,
  "width_mm": 115, "height_mm": 104,
  "stitch_count": 873,
  "color_count": 12,
  "thread_colors": [
    {"index": 0, "name": "Maize", "rgb": [240, 205, 70], "stitch_count": 48},
    {"index": 1, "name": "Crimson", "rgb": [186, 32, 55], "stitch_count": 62}
  ],
  "color_blocks": [
    {"color": 0, "start": 0, "end": 48},
    {"color": 1, "start": 48, "end": 110}
  ],
  "stitches": [
    {"i": 0, "x": 61.32, "y": 59.12, "color": 0},
    {"i": 1, "x": 62.94, "y": 60.86, "color": 0}
  ]
}
```

## How the optimisation works

**Colour reduction (numpy k-means).** The design's 12 thread colours are
treated as points in RGB space, weighted by how many stitches use each
colour. A k-means implementation in pure numpy (deterministic k-means++
seeding with a fixed seed, Lloyd iterations) finds k = 5 cluster centres;
every thread colour is replaced by its nearest centre and stitches are
regrouped, so the reduced design has at most 5 colour blocks. Quality is
reported as the **usage-weighted quantization error** — the sum over
colours of `stitch_count × squared RGB distance to its replacement` — which
is exactly the objective k-means minimises. It is compared against the
naive **top-k-by-usage** baseline (keep the 5 most-used colours, drop the
rest).

**Stitch-path optimisation (NN + 2-opt).** Inside every colour block the
stitches are reordered, so the colour-block structure — and therefore the
thread-change count — is untouched. The greedy **nearest-neighbour** order
is built first (start stitch = most central stitch of the block); then a
**best-improvement 2-opt** local search (vectorised with numpy, one pass =
one row scan of the distance matrix) repeatedly reverses path segments that
shorten the route. The search is run from both the NN order and the
as-digitised order and the better result is kept, so the optimised file is
never worse than the file as shipped. Finally, whole blocks are flipped
where that shortens the jump stitches around them. The reported metric is
**travel distance within colour blocks**, before vs after, in percent.

## Results (measured on the 3 seeded sample designs, k = 5)

| Design | Stitches | Colours | Quant. error k-means | vs top-5 baseline | Thread changes | Path travel (mm) as-digitised → optimised | Path saved |
|---|---|---|---|---|---|---|---|
| rose_bloom | 873 | 12 → 5 | 1,106,128 | −28.6% | 12 → 5 (−58.3%) | 1977.3 → 1920.9 | 2.8% (9.6% vs NN) |
| monogram_a | 644 | 12 → 5 | 748,251 | −74.9% | 12 → 5 (−58.3%) | 1827.1 → 1687.0 | 7.7% (10.0% vs NN) |
| alpine_scene | 685 | 12 → 5 | 892,409 | −83.0% | 12 → 5 (−58.3%) | 2119.8 → 1690.5 | 20.3% (21.7% vs NN) |
| **Total** | **2202** | **36 → 15** | **2,746,788** | **−71.9%** | **36 → 15 (−58.3%)** | **5924.2 → 5298.4** | **10.6% (14.0% vs NN)** |

Headline numbers:

- **71.9% lower colour quantization error** than the top-5-by-usage
  baseline across the three designs, at 5 colours instead of 12.
- **58.3% fewer thread changes** (36 → 15) — ~21 fewer machine stops.
- **10.6% less travel** inside colour blocks versus the as-digitised
  files, and **14.0% versus the nearest-neighbour baseline** (2-opt on top
  of NN); up to 20.3% on a single design.
- Estimated machine-time saving of **~170 s** across the three designs
  (8 s per thread change, 250 mm/s travel — see assumptions below).

These numbers are reproduced by every run: the generator, the k-means
seeding and the NN/2-opt search are all fully deterministic (seed 42).

Machine-readable results: `results/optimization_report.json`.

## Assumptions and limitations

- Time estimate uses 8 s per thread change and 250 mm/s jump-travel speed;
  both are stated constants in `embroidery_optimizer/cli.py` and easy to
  change.
- Colour distance is Euclidean RGB. A perceptually uniform space (CIELAB)
  would be a closer match to human colour judgement, but is not available
  in numpy alone — noted here as future work.
- 2-opt is a local search; it guarantees improvement over its start orders
  but not a global TSP optimum (the problem is NP-hard; exact solvers are
  out of scope for this tool).
- Sample designs are synthetic but structured like real digitised files:
  colour blocks, jump stitches, trims, satin-fill strokes with 2.2 mm
  stitch length.

## Tests

`python tests/test_optimizer.py` — a plain script with asserts, no test
framework. 13 checks: generator determinism and file structure, k-means
determinism/cluster recovery/known-error values, k-means beating the top-k
baseline on all three designs, NN/2-opt correctness (including a strict
improvement case and a never-worsens property test), colour-block
preservation during path optimisation, palette remapping, and end-to-end
CLI smoke tests.
