#!/usr/bin/env python3
"""
Standalone schematic: why coefficient decay along refinement branches tells
us which subdomain needs a bigger expert.

Top row:    a piecewise-constant f(x) on [0, 1] with the tree split at
            x = 0.5 marked. The left subdomain (node A) is a single constant;
            the right one (node B) contains a jump at x = 0.78.
Bottom row: the decision-tree refinement under each node, node shading
            encoding the geometric-wavelet coefficient magnitude.
            Node A: the root carries a coefficient, its two children are
            splits of a constant and carry none (depth 1 shown).
            Node B: the root is large, the split lands on the jump so both
            children carry a similar moderate coefficient, and their
            children (splits of constants) carry none (depth 2 shown).

No repo dependencies, no real data. Usage:

    python scripts/plot_capacity_sketch.py            # -> docs/capacity_sketch.png
    python scripts/plot_capacity_sketch.py out.png    # explicit output path
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle

# ----------------------------------------------------------------------------
# style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.linewidth": 1.6,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

INK = "#1a1a1a"
EDGE = "#8c8c8c"
MUTED = "#555555"
PALE_BLUE = "#e6eef8"
PALE_ORANGE = "#fbeedd"
CMAP = plt.get_cmap("Greys")

# piecewise-constant function: (x0, x1, value)
SEGMENTS = [(0.0, 0.5, 0.25), (0.5, 0.78, 0.72), (0.78, 1.0, 0.52)]
SPLIT = 0.5
JUMP = 0.78


# ----------------------------------------------------------------------------
# top row: the function
# ----------------------------------------------------------------------------
def draw_function(ax):
    ax.add_patch(Rectangle((0.0, 0.0), SPLIT, 1.0, facecolor=PALE_BLUE,
                           edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((SPLIT, 0.0), 1.0 - SPLIT, 1.0, facecolor=PALE_ORANGE,
                           edgecolor="none", zorder=0))

    for x0, x1, v in SEGMENTS:
        ax.plot([x0, x1], [v, v], color=INK, lw=5.0, solid_capstyle="butt",
                zorder=3)

    ax.axvline(SPLIT, color=MUTED, lw=2.0, ls=(0, (5, 4)), zorder=2)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_color(INK)
    ax.spines["left"].set_color(INK)
    ax.set_xlabel(r"$\mathbf{x}$", fontsize=16, color=INK, labelpad=2)
    ax.xaxis.set_label_coords(1.0, -0.04)
    ax.set_ylabel(r"$\mathbf{f(x)}$", fontsize=16, color=INK, rotation=0, labelpad=2)
    ax.yaxis.set_label_coords(-0.005, 1.0)
    ax.yaxis.label.set_ha("right")
    ax.yaxis.label.set_va("top")

    # below the axis: which tree node owns which subdomain, and the split
    ax.text(SPLIT / 2, -0.10, "node A", ha="center", va="top", fontsize=14,
            fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text((SPLIT + 1.0) / 2, -0.10, "node B", ha="center", va="top",
            fontsize=14, fontweight="bold", color=INK, transform=ax.transAxes)
    ax.annotate("tree split", xy=(SPLIT, 0.0), xytext=(SPLIT, -0.10),
                textcoords="axes fraction", ha="center", va="top",
                fontsize=12, fontweight="bold", color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.6,
                                shrinkA=2, shrinkB=0))


# ----------------------------------------------------------------------------
# bottom row: refinement subtrees (decision-tree splits, best threshold each)
# ----------------------------------------------------------------------------
# Schematic wavelet-coefficient magnitudes per level, root first.
#   node A (constant): the root carries the subdomain coefficient; splitting
#                      a constant yields two children with zero coefficient.
#   node B (jump):     the root carries a large coefficient; the best split
#                      lands on the jump, so both children carry a similar,
#                      moderate coefficient (each child is a constant that
#                      differs from the parent mean); splitting those constants
#                      again yields zero coefficients.
TREE_A = [[0.55], [0.03, 0.03]]
TREE_B = [[0.85], [0.45, 0.45], [0.03, 0.03, 0.03, 0.03]]


def draw_tree(ax, levels, header):
    # layout in axes-data coords: x in [0,1], levels from top
    ys = {0: 0.86, 1: 0.55, 2: 0.24}
    pos = {}
    for level, vals in enumerate(levels):
        n = len(vals)
        for i in range(n):
            pos[(level, i)] = ((i + 0.5) / n, ys[level])

    # edges
    for (level, i), (x, y) in pos.items():
        if level == 0:
            continue
        px, py = pos[(level - 1, i // 2)]
        ax.plot([px, x], [py, y], color=EDGE, lw=1.8, zorder=1)

    # nodes
    r = 0.09
    for (level, i), (x, y) in pos.items():
        c = levels[level][i]
        ax.add_patch(Circle((x, y), r, facecolor=CMAP(c), edgecolor=EDGE,
                            lw=1.8, zorder=2))

    ax.text(0.5, 1.0, header, ha="center", va="bottom", fontsize=15,
            fontweight="bold", color=INK, transform=ax.transAxes)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ----------------------------------------------------------------------------
def main(out_path):
    fig = plt.figure(figsize=(9, 6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1.0],
                  hspace=0.45, wspace=0.10,
                  left=0.08, right=0.95, top=0.93, bottom=0.14)

    ax_f = fig.add_subplot(gs[0, :])
    draw_function(ax_f)

    ax_l = fig.add_subplot(gs[1, 0])
    ax_r = fig.add_subplot(gs[1, 1])
    draw_tree(ax_l, TREE_A, "node A's subtree")
    draw_tree(ax_r, TREE_B, "node B's subtree")

    # colour scale for the node fill: wavelet-coefficient norm
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    cax = fig.add_axes([0.38, 0.06, 0.24, 0.025])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([])
    cb.outline.set_edgecolor(EDGE)
    cb.outline.set_linewidth(1.6)
    cax.text(-0.03, 0.5, "0", ha="right", va="center", fontsize=12,
             fontweight="bold", color=INK, transform=cax.transAxes)
    cax.text(1.03, 0.5, "wavelet norm", ha="left", va="center", fontsize=12,
             fontweight="bold", color=INK, transform=cax.transAxes)

    fig.savefig(out_path, dpi=300, facecolor="white")
    print(f"saved {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        out = Path(sys.argv[1])
    else:
        out = Path(__file__).resolve().parent.parent / "docs" / "capacity_sketch.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    main(out)
