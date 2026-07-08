#!/usr/bin/env python3
"""
Standalone figure: the three global-composition variants of the adaptive
tree-of-experts framework — AToE (additive), AToE-Leaves, and ANT.

This reproduces (and cleans up, for paper use) the hand-made schematic figure,
using only numpy + matplotlib. Each of the three panels shows:

    1. DECOMPOSED DOMAIN  — the (x, t) domain split by a depth-2 tree into four
       leaf regions (1-4), each shown as a compact smoothstep window (flat-top
       = 1 inside the box, a C^N collar fading to 0 outside), plus the dashed
       collar/support boundaries and a query point (x, t).
    2. SOFT INDICATORS    — the normalised blending weights Psi~_i(x, t) at that
       query point (AToE normalises over all nodes; AToE-Leaves / ANT over the
       leaves L).
    3. TREE + FLOW        — the tree with each node's weighted contribution
       flowing into the blend Sigma, giving u(x, t). ANT additionally shows the
       parent -> child activation flow (dashed).

Outputs a vector PDF (for LaTeX \includegraphics) and a high-DPI PNG.

Usage:
    python scripts/plot_atoe_variants.py                 # -> AToE/docs/
    python scripts/plot_atoe_variants.py <output_dir>
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, PathPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D

# white halo so ink labels stay legible on top of saturated window colours
HALO = [pe.withStroke(linewidth=2.4, foreground="white")]


# -----------------------------------------------------------------------------
# Palette (matches the design; leaf/region colours are colour-blind safe)
# -----------------------------------------------------------------------------
COL = {"1": "#4477AA", "2": "#228833", "3": "#8E44AD", "4": "#CC3311"}  # leaves
INTERNAL_ACTIVE = "#7E74A8"   # root / L / R when they take part in the blend
INACTIVE = "#C2C8CE"          # root / L / R when retired (leaves-only variants)
INACTIVE_TXT = "#7c848c"
ROOT_BAR = "#5a6169"
ANT_FLOW = "#5a63b0"          # dashed parent -> child activation flow
INK = "#1b1f24"
MUTED = "#6b747e"
AXIS = "#414a54"
LABEL = "#414a54"
GREY_EDGE = "#aeb4ba"
TRACK = "#e3e7eb"
DIVIDER = "#e3e7eb"
BORDER = "#c9ced4"

# leaf region centres in normalised domain coords (x right, t up)
REGION_CENTRE = {
    "1": (0.25, 0.75),   # top-left     (blue)
    "2": (0.75, 0.75),   # top-right    (green)
    "3": (0.25, 0.25),   # bottom-left  (purple)
    "4": (0.75, 0.25),   # bottom-right (red)
}
# leaf region bounds ((x_lo, x_hi), (t_lo, t_hi)) — the four quadrants of the
# unit domain; each expert gets a compact smoothstep window over its box.
REGION_BOUNDS = {
    "1": ((0.0, 0.5), (0.5, 1.0)),
    "2": ((0.5, 1.0), (0.5, 1.0)),
    "3": ((0.0, 0.5), (0.0, 0.5)),
    "4": ((0.5, 1.0), (0.0, 0.5)),
}
COLLAR_ALPHA = 0.2       # collar half-width as a fraction of the box size
WINDOW_ORDER = 2         # smoothstep C^N order (matches adaptive/indicators.py)
# (x, t) query point: placed inside the central overlap [0.4,0.6]^2 where all
# four windows are non-zero, so every leaf weight is genuinely > 0. Because the
# windows factorise, the leaf weights separate as (left/right mass)x(top/bottom
# mass); this spot reproduces ~0.47/0.39/0.08/0.06 for experts 1/2/3/4.
QUERY = (0.47, 0.57)

# tree node positions in the design's local pixel frame (y grows downward);
# we negate y when drawing so matplotlib's y-up frame keeps the same layout.
NODE_PX = {
    "root": (150, 30),
    "L": (86, 108), "R": (214, 108),
    "1": (46, 190), "3": (120, 190), "2": (180, 190), "4": (254, 190),
}
SIGMA_PX = (150, 300)
TREE_EDGES = [("root", "L"), ("root", "R"),
              ("L", "1"), ("L", "3"), ("R", "2"), ("R", "4")]


def _np(px):
    """design pixel coord -> matplotlib coord (y up)."""
    return (px[0], -px[1])


# -----------------------------------------------------------------------------
# Panel definitions
# -----------------------------------------------------------------------------
PANELS = [
    dict(
        tag="(A)", name="AAToE",
        bars=[("root", 0.10), ("L", 0.20), ("R", 0.18),
              ("1", 0.24), ("2", 0.18), ("3", 0.06), ("4", 0.04)],
        contrib={"root": 0.10, "L": 0.20, "R": 0.18,
                 "1": 0.24, "2": 0.18, "3": 0.06, "4": 0.04},
        internal_active=True, ant=False,
        norm="normalised over all nodes",
        eq=r"$u(x,t)=\sum_{i}\ \tilde{\Psi}_i(x,t)\,u_i(x,t)$",
    ),
    dict(
        tag="(B)", name="AToE",
        bars=[("1", 0.46), ("2", 0.39), ("3", 0.09), ("4", 0.06)],
        contrib={"1": 0.46, "2": 0.39, "3": 0.09, "4": 0.06},
        internal_active=False, ant=False,
        norm=r"normalised over leaves $\mathcal{L}$",
        eq=r"$u(x,t)=\sum_{i\in\mathcal{L}}\ \tilde{\Psi}_i(x,t)\,u_i(x,t)$",
    ),
    dict(
        tag="(C)", name="ANT",
        bars=[("1", 0.46), ("2", 0.39), ("3", 0.09), ("4", 0.06)],
        contrib={"1": 0.46, "2": 0.39, "3": 0.09, "4": 0.06},
        internal_active=False, ant=True,
        norm=r"normalised over leaves $\mathcal{L}$",
        eq=r"$u(x,t)=\sum_{i\in\mathcal{L}}\ \tilde{\Psi}_i(x,t)\,u_i(x,t)$",
    ),
]


def node_facecolor(name, internal_active):
    if name in ("root", "L", "R"):
        return INTERNAL_ACTIVE if internal_active else INACTIVE
    return COL[name]


def flow_lw(value):
    """Weighted contribution -> flow stroke width."""
    return 1.2 + 8.5 * value


# -----------------------------------------------------------------------------
# 1) Decomposed-domain panel
# -----------------------------------------------------------------------------
def _compact_ramp(s, N=WINDOW_ORDER):
    """One-sided smoothstep ramp: 0 for s<=0, S_N in (0,1), 1 for s>=1."""
    s = np.clip(s, 0.0, 1.0)
    if N == 1:
        return 3 * s**2 - 2 * s**3
    if N == 2:
        return 6 * s**5 - 15 * s**4 + 10 * s**3
    if N == 3:
        return 35 * s**4 - 84 * s**5 + 70 * s**6 - 20 * s**7
    return 126 * s**5 - 420 * s**6 + 540 * s**7 - 315 * s**8 + 70 * s**9


def region_window(X, T, bounds, alpha=COLLAR_ALPHA, N=WINDOW_ORDER):
    """Compact smoothstep window: 1 inside the box, C^N collar, exact 0 beyond.

    Mirrors adaptive.indicators.SoftIndicator (window_type='smoothstep').
    """
    (a_x, b_x), (a_t, b_t) = bounds
    dx, dt = alpha * (b_x - a_x), alpha * (b_t - a_t)
    wx = _compact_ramp((X - (a_x - dx)) / dx) * _compact_ramp(((b_x + dx) - X) / dx)
    wt = _compact_ramp((T - (a_t - dt)) / dt) * _compact_ramp(((b_t + dt) - T) / dt)
    return wx * wt


def domain_windows(res=720, amp=0.62):
    """Composite the four smoothstep windows as a partition of unity over white.

    Inside a box the colour is flat (window = 1); across the collars neighbouring
    windows overlap and blend, exactly as the normalised indicators do.
    """
    g = np.linspace(0.0, 1.0, res)
    X, T = np.meshgrid(g, g)                       # origin='lower' -> T up
    keys = list(REGION_BOUNDS)
    W = np.stack([region_window(X, T, REGION_BOUNDS[k]) for k in keys], axis=0)
    cols = np.stack([np.array(to_rgb(COL[k])) for k in keys], axis=0)   # (4, 3)
    total = W.sum(axis=0) + 1e-9
    colour = np.tensordot(W / total, cols, axes=([0], [0]))             # (res, res, 3)
    a = (amp * np.clip(W.sum(axis=0), 0.0, 1.0))[..., None]
    return colour * a + 1.0 * (1.0 - a)


def draw_domain(ax):
    ax.set_xlim(-0.13, 1.13)
    ax.set_ylim(-0.13, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.imshow(domain_windows(), extent=[0, 1, 0, 1], origin="lower",
              interpolation="bilinear", zorder=0)

    # clip all decorations to the unit square
    unit = MplPath([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)],
                   [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO,
                    MplPath.LINETO, MplPath.CLOSEPOLY])

    def clipped(artist):
        artist.set_clip_path(unit, ax.transData)
        return artist

    # domain boundary (dashed)
    ax.add_patch(clipped(Rectangle(
        (0.02, 0.02), 0.96, 0.96, fill=False, ls=(0, (7, 4)),
        ec=INK, alpha=0.6, lw=1.0, zorder=2)))
    # region (flat-top) boundaries: the 2x2 tiling
    for xy in ([[0.5, 0.5], [0, 1]], [[0, 1], [0.5, 0.5]]):
        ax.add_line(clipped(Line2D(xy[0], xy[1], color="white", lw=1.4,
                                   alpha=0.85, zorder=2)))
    # per-region collar (support) boundaries — dashed squares, Psi~ -> 0 outside
    for k, ((a_x, b_x), (a_t, b_t)) in REGION_BOUNDS.items():
        dx, dt = COLLAR_ALPHA * (b_x - a_x), COLLAR_ALPHA * (b_t - a_t)
        ax.add_patch(clipped(Rectangle(
            (a_x - dx, a_t - dt), (b_x - a_x) + 2 * dx, (b_t - a_t) + 2 * dt,
            fill=False, ec=COL[k], lw=1.1, alpha=0.7,
            ls=(0, (4, 3)), zorder=2)))
    # region centre labels
    for k, (cx, cy) in REGION_CENTRE.items():
        ax.text(cx, cy, k, color=INK, fontsize=13, fontweight="bold",
                ha="center", va="center", zorder=3, path_effects=HALO)

    # L / R / root text markers
    ax.text(0.06, 0.5, "L", color=INK, fontsize=11, fontweight="bold",
            ha="center", va="center", zorder=3, path_effects=HALO)
    ax.text(0.94, 0.5, "R", color=INK, fontsize=11, fontweight="bold",
            ha="center", va="center", zorder=3, path_effects=HALO)
    ax.text(0.055, 0.935, "root", color=INK, fontsize=8.5, fontweight="bold",
            ha="left", va="center", zorder=3, path_effects=HALO)

    # query point (x, t)
    qx, qy = QUERY
    for art in (ax.plot([qx - 0.03, qx + 0.03], [qy, qy], color=INK, lw=1.6, zorder=4)
                + ax.plot([qx, qx], [qy - 0.03, qy + 0.03], color=INK, lw=1.6, zorder=4)
                + ax.plot(qx, qy, "o", color=INK, ms=4, zorder=4)):
        art.set_path_effects(HALO)
    ax.text(qx + 0.05, qy + 0.05, r"$(x,t)$", color=INK, fontsize=11,
            style="italic", ha="left", va="bottom", zorder=4, path_effects=HALO)

    # axes arrows
    ax.annotate("", xy=(1.07, -0.03), xytext=(0.0, -0.03),
                arrowprops=dict(arrowstyle="-|>", color=AXIS, lw=1.1))
    ax.annotate("", xy=(-0.03, 1.07), xytext=(-0.03, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=AXIS, lw=1.1))
    ax.text(1.09, -0.03, "$x$", color=AXIS, fontsize=12, ha="left", va="center")
    ax.text(-0.03, 1.10, "$t$", color=AXIS, fontsize=12, ha="center", va="bottom")
    ax.text(-0.055, -0.055, "0", color=MUTED, fontsize=9, ha="center", va="center")


# -----------------------------------------------------------------------------
# 2) Soft-indicator bar panel
# -----------------------------------------------------------------------------
def draw_bars(ax, panel):
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 8.6)
    ax.axis("off")

    y_top, pitch = 6.5, 0.95
    x0, x1 = 0.15, 0.80          # indicator track span

    # header
    ax.text(0.0, 7.7, r"soft indicators $\tilde{\Psi}_i(x,t)$",
            color=INK, fontsize=13, ha="left", va="center")
    ax.text(1.0, 7.7, r"$\Sigma = 1.00$", color=INK, fontsize=13,
            ha="right", va="center")

    for i, (name, val) in enumerate(panel["bars"]):
        y = y_top - i * pitch
        swatch = (ROOT_BAR if name == "root"
                  else INTERNAL_ACTIVE if name in ("L", "R") else COL[name])
        ax.plot(0.015, y, marker="s", ms=7, color=swatch, zorder=3)
        ax.text(0.06, y, name, color=LABEL, fontsize=9.5, fontweight="bold",
                ha="left", va="center")
        ax.plot([x0, x1], [y, y], color=TRACK, lw=5.5,
                solid_capstyle="round", zorder=2)
        ax.plot([x0, x0 + (x1 - x0) * val], [y, y], color=swatch, lw=5.5,
                solid_capstyle="round", zorder=3)
        ax.text(1.0, y, f"{val:.2f}", color=LABEL, fontsize=10,
                ha="right", va="center")


# -----------------------------------------------------------------------------
# 3) Tree + flow panel
# -----------------------------------------------------------------------------
def _flow_path(p0, p1):
    (x0, y0), (x1, y1) = p0, p1
    ym = 0.5 * (y0 + y1)
    verts = [(x0, y0), (x0, ym), (x1, ym), (x1, y1)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    return MplPath(verts, codes)


def _ant_arrow(ax, p_parent, p_child, off=8.0):
    """Dashed activation-flow arrow parallel to a parent->child edge."""
    (px, py), (cx, cy) = p_parent, p_child
    dx, dy = cx - px, cy - py
    n = np.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    perp = (-uy, ux)
    sx, sy = px + perp[0] * off, py + perp[1] * off
    ex, ey = cx + perp[0] * off, cy + perp[1] * off
    ax.add_patch(FancyArrowPatch(
        (sx + ux * 22, sy + uy * 22), (ex - ux * 20, ey - uy * 20),
        arrowstyle="-|>", mutation_scale=9, lw=1.5, ls="--",
        color=ANT_FLOW, zorder=2.2))


def draw_tree(ax, panel):
    ax.set_xlim(-8, 308)
    ax.set_ylim(-352, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    active = panel["internal_active"]
    contrib = panel["contrib"]
    sigma = _np(SIGMA_PX)

    # structural edges
    for a, b in TREE_EDGES:
        pa, pb = _np(NODE_PX[a]), _np(NODE_PX[b])
        ax.add_line(Line2D([pa[0], pb[0]], [pa[1], pb[1]],
                           color=GREY_EDGE, lw=1.3, zorder=1))

    # ANT activation flow (dashed, parent -> child)
    if panel["ant"]:
        for a, b in TREE_EDGES:
            _ant_arrow(ax, _np(NODE_PX[a]), _np(NODE_PX[b]))

    # contribution flows -> Sigma
    for name, val in contrib.items():
        colr = node_facecolor(name, active)
        p0 = (_np(NODE_PX[name])[0], _np(NODE_PX[name])[1] - 16)
        p1 = (sigma[0], sigma[1] + 17)
        ax.add_patch(PathPatch(_flow_path(p0, p1), fill=False, ec=colr,
                               lw=flow_lw(val), alpha=min(0.95, 0.5 + 0.9 * val),
                               capstyle="round", zorder=2.5))

    # internal nodes (root / L / R) as rounded squares
    for name in ("root", "L", "R"):
        cx, cy = _np(NODE_PX[name])
        fc = node_facecolor(name, active)
        tc = "white" if active else INACTIVE_TXT
        ax.add_patch(FancyBboxPatch(
            (cx - 15, cy - 15), 30, 30,
            boxstyle="round,pad=0,rounding_size=6",
            fc=fc, ec="white", lw=1.6,
            alpha=0.92 if active else 0.85, zorder=3))
        fs = 8 if name == "root" else 10.5
        ax.text(cx, cy, name, color=tc, fontsize=fs, fontweight="bold",
                ha="center", va="center", zorder=4)

    # leaf nodes as circles (minor leaves faded)
    for name in ("1", "2", "3", "4"):
        cx, cy = _np(NODE_PX[name])
        alpha = 1.0 if contrib.get(name, 0) >= 0.15 else 0.55
        ax.add_patch(Circle((cx, cy), 15.5, fc=COL[name], ec="white",
                            lw=1.6, alpha=alpha, zorder=3))
        ax.text(cx, cy, name, color="white", fontsize=12.5, fontweight="bold",
                ha="center", va="center", zorder=4)

    # Sigma blend node
    ax.add_patch(Circle(sigma, 17, fc="white", ec=INK, lw=1.5, zorder=4))
    ax.text(sigma[0], sigma[1], r"$\Sigma$", color=INK, fontsize=17,
            ha="center", va="center", zorder=5)

    # input (x, t) into the root
    ax.annotate("", xy=(150, -13), xytext=(150, -2),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(150, 4, r"$(x,t)$", color=INK, fontsize=12, style="italic",
            ha="center", va="bottom")
    # Sigma -> u(x, t)
    ax.annotate("", xy=(150, -332), xytext=(150, -318),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(150, -344, r"$u(x,t)$", color=INK, fontsize=12.5, style="italic",
            ha="center", va="top")


# -----------------------------------------------------------------------------
# Assemble the full figure
# -----------------------------------------------------------------------------
def build_figure():
    fig = plt.figure(figsize=(13.0, 10.6))

    # background axes for borders + dividers
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    bg.axis("off")
    bg.add_patch(Rectangle((0.004, 0.004), 0.992, 0.992, fill=False,
                           ec=BORDER, lw=1.0))

    col_w, left0, gap = 0.28, 0.05, 0.035
    lefts = [left0 + i * (col_w + gap) for i in range(3)]
    centres = [lft + col_w / 2 for lft in lefts]

    # column dividers
    for xd in (lefts[1] - gap / 2, lefts[2] - gap / 2):
        bg.add_line(Line2D([xd, xd], [0.045, 0.945], color=DIVIDER, lw=1.0))

    for panel, lft, cx in zip(PANELS, lefts, centres):
        # column headings
        fig.text(lft, 0.958, panel["tag"], color=MUTED, fontsize=13,
                 fontweight="bold", ha="left", va="center")
        fig.text(lft + 0.028, 0.958, panel["name"], color=INK, fontsize=13,
                 fontweight="bold", ha="left", va="center")
        fig.text(lft, 0.937, "D E C O M P O S E D   D O M A I N", color=MUTED,
                 fontsize=8, fontweight="bold", ha="left", va="center")

        ax_dom = fig.add_axes([lft, 0.635, col_w, 0.295])
        draw_domain(ax_dom)

        ax_bar = fig.add_axes([lft, 0.475, col_w, 0.14])
        draw_bars(ax_bar, panel)

        ax_tree = fig.add_axes([lft, 0.135, col_w, 0.32])
        draw_tree(ax_tree, panel)

        fig.text(cx, 0.085, panel["eq"], color=INK, fontsize=15,
                 ha="center", va="center")

    return fig


def main(out_dir=None):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
    })

    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "docs"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = build_figure()
    stem = "atoe_variants"
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    return pdf_path, png_path


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
