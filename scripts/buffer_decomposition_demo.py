"""Gradient-driven buffer regions between neighbouring experts (exploratory).

Takes an existing "perfect tree" decomposition (leaves tile the domain
exactly) and, for every interior interface between two leaves, grows a
BUFFER band across it. The band's two edges are placed independently by
marching outward from the interface until the mean directional gradient of
the solution ON THAT FACE drops below a global quantile of the same
gradient field.

Sensor (per dimension z, over the whole domain):

    G_z(x, t) = |d u / d z|          (L2 over output channels)
    ref_z(q)  = quantile_q( G_z )

For an interface at z = z0 shared by a "low" leaf and a "high" leaf over
the co-dimension interval [o_lo, o_hi], the face statistic is

    s(z) = mean_{o in [o_lo, o_hi]} G_z(z, o)

and the buffer edge on each side is the FIRST face, marching outward from
z0, with s(z) < ref_z(q). If no such face exists inside the cap, the
argmin of s over the scan window is used instead, so the growth always
terminates. Each side is capped at ``--max-frac`` of that side's own
extent in z, so a buffer can never reach a second interface, and is
clipped to the domain box.

Low q  = strict bar = wide buffers.  High q = loose bar = thin buffers.

Each figure has two rows over the ground-truth heatmap. The top row
sweeps the quantile q and shows the leaf boxes (black) with the grown
buffer bands (red). The bottom row is the CURRENT method for comparison:
the same leaves with the fixed smoothstep collar of Section 3.6, swept
over sigma_fraction. There the collar half-width is
delta_j = sigma_fraction * (b_j - a_j) per dimension, so every region's
support grows to [a_j - delta_j, b_j + delta_j]; the red frame is that
added collar. Collars are clipped to the domain box so both rows share
axis limits and read on the same scale.

Ground truth comes from each solver's native grid (the same source the
perfect trees were fit on), so the picture is the "oracle" version of the
idea; the deployed method would run the identical sensor on the root
prediction u_0.

Usage:
    python scripts/buffer_decomposition_demo.py
    python scripts/buffer_decomposition_demo.py --quantiles 0.8 0.9 \
        --problems burgers1d kdv --out outputs/buffer_demo
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.utils import native_ground_truth_grid  # noqa: E402
from utils.plot_io import save_png  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TREES = (REPO / 'perfect_tree_examples' / 'M_8_e_0.0'
                 / 'perfect_trees.json')
DIM_NAMES = ('x', 't')
TOL = 1e-9
# Output components per multi-output problem, in the order the solver
# stacks them (trainer.utils.native_ground_truth_grid: real, then imag).
CHANNEL_NAMES = {'schrodinger': ('u', 'v')}


# --------------------------------------------------------------------------
# sensor
# --------------------------------------------------------------------------

def gradient_fields(gt_grid, grid_x, grid_t):
    """Per-dimension gradient magnitude on the native grid.

    ``gt_grid`` is (nx, nt) for scalar problems or (nx, nt, C) for
    multi-output ones; channels are reduced with an L2 norm so the sensor
    is a single non-negative field per dimension.

    Returns ``[G_x, G_t]``, each (nx, nt).
    """
    u = gt_grid if gt_grid.ndim == 3 else gt_grid[:, :, None]
    per_dim = []
    for axis, coord in ((0, grid_x), (1, grid_t)):
        chans = [np.gradient(u[:, :, c], coord, axis=axis)
                 for c in range(u.shape[2])]
        per_dim.append(np.sqrt(np.sum(np.square(chans), axis=0)))
    return per_dim


def face_mean(g, idx, dim, o_mask):
    """Mean of sensor ``g`` on the face at grid index ``idx`` normal to ``dim``.

    ``o_mask`` is a boolean mask over the co-dimension's grid coordinates,
    selecting the part of the face the two leaves actually share.
    """
    face = g[idx, :] if dim == 0 else g[:, idx]
    return float(face[o_mask].mean())


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def find_interfaces(leaves):
    """All shared interior faces between leaf pairs.

    Returns dicts with the split dimension, its coordinate, the shared
    extent in the other dimension, and the two adjacent leaves (``low``
    lies below ``z0``, ``high`` above it).
    """
    out = []
    for a, b in combinations(leaves, 2):
        for dim in (0, 1):
            o = 1 - dim
            o_lo = max(a['bounds_lower'][o], b['bounds_lower'][o])
            o_hi = min(a['bounds_upper'][o], b['bounds_upper'][o])
            if o_hi - o_lo <= TOL:
                continue          # corner touch, not a shared face
            if abs(a['bounds_upper'][dim] - b['bounds_lower'][dim]) < TOL:
                low, high = a, b
            elif abs(b['bounds_upper'][dim] - a['bounds_lower'][dim]) < TOL:
                low, high = b, a
            else:
                continue
            out.append({
                'dim': dim,
                'z0': low['bounds_upper'][dim],
                'o_lo': o_lo,
                'o_hi': o_hi,
                'low': low,
                'high': high,
            })
    return out


# --------------------------------------------------------------------------
# buffer construction
# --------------------------------------------------------------------------

def build_buffers(leaves, domain, grads, grid, q, min_frac, max_frac):
    """Grow a buffer band across every interior interface.

    Returns ``(buffers, refs)`` where each buffer carries its box, the
    interface it straddles, and whether each edge actually met the
    threshold (vs. falling back to the local argmin).
    """
    coords = list(grid)
    refs = [float(np.quantile(g, q)) for g in grads]

    buffers = []
    for itf in find_interfaces(leaves):
        dim, o = itf['dim'], 1 - itf['dim']
        g, z_coord, o_coord = grads[dim], coords[dim], coords[1 - itf['dim']]
        z0, ref = itf['z0'], refs[dim]

        o_mask = ((o_coord >= itf['o_lo'] - TOL)
                  & (o_coord <= itf['o_hi'] + TOL))
        if not o_mask.any():                    # face thinner than a cell
            o_mask = np.zeros_like(o_coord, dtype=bool)
            o_mask[np.argmin(np.abs(o_coord - 0.5 * (itf['o_lo']
                                                     + itf['o_hi'])))] = True

        edges, hits = {}, {}
        for side, leaf, sign in (('lo', itf['low'], -1),
                                 ('hi', itf['high'], +1)):
            span = leaf['bounds_upper'][dim] - leaf['bounds_lower'][dim]
            near = z0 + sign * min_frac * span
            far = z0 + sign * max_frac * span
            far = min(max(far, domain['lower'][dim]), domain['upper'][dim])

            lo_b, hi_b = sorted((near, far))
            cand = np.nonzero((z_coord >= lo_b - TOL)
                              & (z_coord <= hi_b + TOL))[0]
            if cand.size == 0:
                edges[side], hits[side] = near, False
                continue
            cand = cand[np.argsort(np.abs(z_coord[cand] - z0))]

            stats = np.array([face_mean(g, int(i), dim, o_mask)
                              for i in cand])
            below = np.nonzero(stats < ref)[0]
            if below.size:
                pick, hits[side] = cand[below[0]], True
            else:
                pick, hits[side] = cand[int(np.argmin(stats))], False
            edges[side] = float(z_coord[pick])

        lower, upper = [0.0, 0.0], [0.0, 0.0]
        lower[dim], upper[dim] = edges['lo'], edges['hi']
        lower[o], upper[o] = itf['o_lo'], itf['o_hi']

        buffers.append({
            'dim': dim,
            'interface_coord': z0,
            'bounds_lower': lower,
            'bounds_upper': upper,
            'width': upper[dim] - lower[dim],
            'width_lo': z0 - edges['lo'],
            'width_hi': edges['hi'] - z0,
            'threshold_met_lo': hits['lo'],
            'threshold_met_hi': hits['hi'],
        })
    return buffers, refs


def build_region_tiling(leaves, buffers, domain):
    """Partition the domain into regions, with buffers counted as regions.

    Cuts the domain on every leaf and buffer coordinate, then labels each
    atomic cell by the SET of buffers covering it, falling back to the
    containing leaf where no buffer covers it. Cells sharing a label are
    one region, so a buffer band is a single region straddling its
    interface, and the corner where an x-buffer crosses a t-buffer becomes
    a region of its own. The result tiles the domain by construction.

    Returns ``(red_segments, black_segments, n_regions)``. A segment is
    red when it bounds a buffer region, black when it is an original leaf
    interface that no buffer touched.
    """
    live = [b for b in buffers if b['width'] > TOL]

    cuts = []
    for d in (0, 1):
        vals = {domain['lower'][d], domain['upper'][d]}
        for box in list(leaves) + live:
            vals.add(box['bounds_lower'][d])
            vals.add(box['bounds_upper'][d])
        v = np.array(sorted(vals))
        cuts.append(v[np.concatenate([[True], np.diff(v) > TOL])])
    xs, ts = cuts

    CX, CT = np.meshgrid(0.5 * (xs[:-1] + xs[1:]),
                         0.5 * (ts[:-1] + ts[1:]), indexing='ij')

    def _inside(box):
        lo, hi = box['bounds_lower'], box['bounds_upper']
        return ((CX > lo[0]) & (CX < hi[0])
                & (CT > lo[1]) & (CT < hi[1]))

    stack = (np.stack([_inside(b) for b in live]) if live
             else np.zeros((0,) + CX.shape, dtype=bool))
    leaf_id = np.full(CX.shape, -1, dtype=np.int64)
    for i, leaf in enumerate(leaves):
        leaf_id[_inside(leaf)] = i

    key_to_id = {}
    labels = np.empty(CX.shape, dtype=np.int64)
    is_buf = stack.any(axis=0) if live else np.zeros(CX.shape, dtype=bool)
    for i in range(CX.shape[0]):
        for j in range(CX.shape[1]):
            covering = tuple(np.nonzero(stack[:, i, j])[0]) if live else ()
            key = ('buf', covering) if covering \
                else ('leaf', int(leaf_id[i, j]))
            labels[i, j] = key_to_id.setdefault(key, len(key_to_id))

    red, black = [], []
    for i in range(CX.shape[0] - 1):          # faces normal to x
        for j in range(CX.shape[1]):
            if labels[i, j] != labels[i + 1, j]:
                seg = [(xs[i + 1], ts[j]), (xs[i + 1], ts[j + 1])]
                bucket = red if (is_buf[i, j] or is_buf[i + 1, j]) else black
                bucket.append(seg)
    for i in range(CX.shape[0]):              # faces normal to t
        for j in range(CX.shape[1] - 1):
            if labels[i, j] != labels[i, j + 1]:
                seg = [(xs[i], ts[j + 1]), (xs[i + 1], ts[j + 1])]
                bucket = red if (is_buf[i, j] or is_buf[i, j + 1]) else black
                bucket.append(seg)

    return red, black, len(key_to_id)


def collar_strips(leaves, sigma, domain):
    """The added collar of the CURRENT method, as disjoint boxes.

    Section 3.6 grows each region's support to [a_j - d_j, b_j + d_j] with
    d_j = sigma * (b_j - a_j) per dimension. The added material is the
    frame between that support and the flat-top core, emitted here as four
    non-overlapping strips per leaf so alpha shading and coverage counts
    reflect genuine overlap between neighbours rather than self-overlap.
    Strips are clipped to the domain box.
    """
    out = []
    for leaf in leaves:
        a, b = leaf['bounds_lower'], leaf['bounds_upper']
        d = [sigma * (b[j] - a[j]) for j in (0, 1)]
        o_lo = [max(a[j] - d[j], domain['lower'][j]) for j in (0, 1)]
        o_hi = [min(b[j] + d[j], domain['upper'][j]) for j in (0, 1)]
        for lower, upper in (
            ([o_lo[0], o_lo[1]], [o_hi[0], a[1]]),      # below the core
            ([o_lo[0], b[1]], [o_hi[0], o_hi[1]]),      # above the core
            ([o_lo[0], a[1]], [a[0], b[1]]),            # left of the core
            ([b[0], a[1]], [o_hi[0], b[1]]),            # right of the core
        ):
            if upper[0] - lower[0] > TOL and upper[1] - lower[1] > TOL:
                out.append({'bounds_lower': lower, 'bounds_upper': upper})
    return out


def coverage_stats(boxes, grid_x, grid_t):
    """Fraction of the domain inside >=1 box, and mean stacking depth."""
    XX, TT = np.meshgrid(grid_x, grid_t, indexing='ij')
    count = np.zeros_like(XX, dtype=np.int32)
    for b in boxes:
        lo, hi = b['bounds_lower'], b['bounds_upper']
        count += ((XX >= lo[0]) & (XX <= hi[0])
                  & (TT >= lo[1]) & (TT <= hi[1])).astype(np.int32)
    return {
        'covered_fraction': float((count > 0).mean()),
        'mean_overlap_depth': float(count.mean()),
        'max_overlap_depth': int(count.max()) if count.size else 0,
    }


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

def draw_heatmap(ax, gt_grid, grid_x, grid_t, alpha=1.0, channel=None):
    """Field background. ``channel=None`` shows the magnitude over outputs.

    A single output channel is signed, so it gets a diverging map centred
    on zero; the magnitude keeps viridis.
    """
    XX, TT = np.meshgrid(grid_x, grid_t, indexing='ij')
    if channel is None:
        display = (np.linalg.norm(gt_grid, axis=2) if gt_grid.ndim == 3
                   else gt_grid)
        kw = {'cmap': 'viridis'}
    else:
        display = gt_grid[:, :, channel]
        lim = float(np.abs(display).max())
        kw = {'cmap': 'RdBu_r', 'vmin': -lim, 'vmax': lim}
    ax.pcolormesh(XX, TT, display, shading='auto', alpha=alpha, zorder=0,
                  **kw)


def _legend(ax, handles):
    ax.legend(handles=handles, loc='upper left', fontsize=7.5,
              framealpha=0.9, handlelength=1.6, borderpad=0.5).set_zorder(20)


def draw_panel(ax, leaves, buffers, domain, gt_grid, grid_x, grid_t, title,
               legend=False, channel=None):
    draw_heatmap(ax, gt_grid, grid_x, grid_t, channel=channel)

    for b in buffers:
        lo, hi = b['bounds_lower'], b['bounds_upper']
        w, h = hi[0] - lo[0], hi[1] - lo[1]
        if w <= 0 or h <= 0:
            continue
        # Hatched where an edge fell back to the local argmin instead of
        # actually crossing the threshold, i.e. the growth hit its cap.
        capped = not (b['threshold_met_lo'] and b['threshold_met_hi'])
        ax.add_patch(patches.Rectangle(
            (lo[0], lo[1]), w, h, linewidth=0.0, facecolor='#ffffff',
            alpha=0.45, zorder=4))
        ax.add_patch(patches.Rectangle(
            (lo[0], lo[1]), w, h, linewidth=1.0, facecolor='#d62728',
            alpha=0.28, hatch='///' if capped else None,
            edgecolor='#8c0d0d', zorder=5))

    for leaf in leaves:
        lo, hi = leaf['bounds_lower'], leaf['bounds_upper']
        ax.add_patch(patches.Rectangle(
            (lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
            linewidth=1.1, edgecolor='black', facecolor='none', zorder=10))

    xr = domain['upper'][0] - domain['lower'][0]
    tr = domain['upper'][1] - domain['lower'][1]
    ax.set_xlim(domain['lower'][0] - 0.03 * xr, domain['upper'][0] + 0.03 * xr)
    ax.set_ylim(domain['lower'][1] - 0.03 * tr, domain['upper'][1] + 0.03 * tr)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title(title, fontsize=10)

    if legend:
        _legend(ax, [
            patches.Patch(fill=False, edgecolor='black',
                          label='leaf region (from the tree)'),
            patches.Patch(facecolor='#d62728', alpha=0.45,
                          edgecolor='#8c0d0d',
                          label='buffer — both edges met the threshold'),
            patches.Patch(facecolor='#d62728', alpha=0.45, hatch='///',
                          edgecolor='#8c0d0d',
                          label='buffer — an edge hit its cap, so it fell\n'
                                'back to the local argmin (width not set\n'
                                'by the sensor)'),
        ])


def draw_collar_panel(ax, leaves, strips, domain, gt_grid, grid_x, grid_t,
                      title, legend=False, channel=None):
    """The current method's fixed smoothstep collar, for comparison."""
    draw_heatmap(ax, gt_grid, grid_x, grid_t, channel=channel)
    for s in strips:
        lo, hi = s['bounds_lower'], s['bounds_upper']
        w, h = hi[0] - lo[0], hi[1] - lo[1]
        ax.add_patch(patches.Rectangle(
            (lo[0], lo[1]), w, h, linewidth=0.0, facecolor='#ffffff',
            alpha=0.45, zorder=4))
        ax.add_patch(patches.Rectangle(
            (lo[0], lo[1]), w, h, linewidth=0.0, facecolor='#d62728',
            alpha=0.28, zorder=5))

    for leaf in leaves:
        lo, hi = leaf['bounds_lower'], leaf['bounds_upper']
        ax.add_patch(patches.Rectangle(
            (lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
            linewidth=1.1, edgecolor='black', facecolor='none', zorder=10))

    xr = domain['upper'][0] - domain['lower'][0]
    tr = domain['upper'][1] - domain['lower'][1]
    ax.set_xlim(domain['lower'][0] - 0.03 * xr, domain['upper'][0] + 0.03 * xr)
    ax.set_ylim(domain['lower'][1] - 0.03 * tr, domain['upper'][1] + 0.03 * tr)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title(title, fontsize=10)

    if legend:
        _legend(ax, [
            patches.Patch(fill=False, edgecolor='black',
                          label='leaf region (flat top of the window)'),
            patches.Patch(facecolor='#d62728', alpha=0.45,
                          label='collar added by sigma_fraction'),
        ])


def background_variants(problem, gt_grid):
    """Which field backgrounds to render, as ``(channel, suffix, label)``.

    Scalar problems get one figure. Multi-output problems additionally get
    one figure per output component on top of the magnitude, so the
    decomposition can be judged against each component's own structure —
    the geometry is identical across all of them.
    """
    variants = [(None, '', 'magnitude |h|' if gt_grid.ndim == 3 else 'u')]
    if gt_grid.ndim == 3:
        names = CHANNEL_NAMES.get(
            problem, tuple(f'c{c}' for c in range(gt_grid.shape[2])))
        for c in range(gt_grid.shape[2]):
            name = names[c] if c < len(names) else f'c{c}'
            variants.append((c, f'_{name}', f'{name} component'))
    return variants


def process_problem(problem, tree, base_cfg, quantiles, sigmas, min_frac,
                    max_frac, out_dir):
    leaves = [n for n in tree['accepted_nodes_bfs']
              if n['is_leaf_in_pruned_tree']]
    domain = tree['domain_bounds']

    cfg = dict(base_cfg)
    cfg['problem'] = problem
    native = native_ground_truth_grid(cfg, max_points_per_axis=1024)
    if native is None:
        print(f"  {problem}: no native solver grid, skipped")
        return None
    gt_grid, grid_x, grid_t = native
    grads = gradient_fields(gt_grid, grid_x, grid_t)

    print(f"  {problem}: {len(leaves)} leaves, grid "
          f"{len(grid_x)}x{len(grid_t)}, "
          f"mean|du/dx|={grads[0].mean():.4g}, "
          f"mean|du/dt|={grads[1].mean():.4g}")

    record = {'problem': problem, 'domain_bounds': domain,
              'n_leaves': len(leaves), 'collar_sweep': [], 'sweep': []}

    # Geometry is independent of which field we draw behind it, so both
    # sweeps are computed once and every background variant reuses them.
    collars = []
    for sigma in sigmas:
        strips = collar_strips(leaves, sigma, domain)
        s = coverage_stats(strips, grid_x, grid_t)
        collars.append((sigma, strips, s))
        record['collar_sweep'].append({'sigma_fraction': sigma, **s})
        print(f"    sigma={sigma:<5} collar covers "
              f"{s['covered_fraction'] * 100:5.1f}%  "
              f"mean depth {s['mean_overlap_depth']:.2f}")

    sweeps = []
    for q in quantiles:
        buffers, refs = build_buffers(leaves, domain, grads,
                                      (grid_x, grid_t), q, min_frac, max_frac)
        stats = coverage_stats(buffers, grid_x, grid_t)
        widths = [b['width'] for b in buffers]
        n_capped = sum(1 for b in buffers
                       if not (b['threshold_met_lo'] and b['threshold_met_hi']))
        _, _, n_regions = build_region_tiling(leaves, buffers, domain)
        sweeps.append((q, buffers, stats, n_capped))

        record['sweep'].append({
            'quantile': q,
            'ref_dx': refs[0], 'ref_dt': refs[1],
            'n_buffers': len(buffers),
            'n_capped': n_capped,
            'n_regions': n_regions,
            'mean_width': float(np.mean(widths)) if widths else 0.0,
            'max_width': float(np.max(widths)) if widths else 0.0,
            **stats,
            'buffers': buffers,
        })
        print(f"    q={q:.2f}  ref=({refs[0]:.4g}, {refs[1]:.4g})  "
              f"{len(buffers)} buffers ({n_capped} capped)  "
              f"covered {stats['covered_fraction'] * 100:5.1f}%  "
              f"mean width {np.mean(widths) if widths else 0:.4f}  "
              f"-> {n_regions} regions")

    ncol = 1 + max(len(quantiles), len(sigmas))
    for channel, suffix, label in background_variants(problem, gt_grid):
        fig, axes = plt.subplots(2, ncol, figsize=(5.2 * ncol, 9.4))
        draw_panel(axes[0, 0], leaves, [], domain, gt_grid, grid_x, grid_t,
                   f'{problem} — background: {label}\n'
                   f'base decomposition ({len(leaves)} leaves)',
                   legend=True, channel=channel)
        for ax in axes[0, len(quantiles) + 1:]:
            ax.axis('off')
        axes[1, 0].axis('off')
        for ax in axes[1, len(sigmas) + 1:]:
            ax.axis('off')

        for col, (q, buffers, stats, n_capped) in enumerate(sweeps, start=1):
            draw_panel(
                axes[0, col], leaves, buffers, domain, gt_grid, grid_x, grid_t,
                f'q = {q:.2f}   |   {len(buffers)} buffers, '
                f'{n_capped} capped\n'
                f'domain covered {stats["covered_fraction"] * 100:.1f}%   '
                f'mean depth {stats["mean_overlap_depth"]:.2f}',
                channel=channel)

        for col, (sigma, strips, s) in enumerate(collars, start=1):
            draw_collar_panel(
                axes[1, col], leaves, strips, domain, gt_grid, grid_x, grid_t,
                f'CURRENT method   |   sigma_fraction = {sigma}\n'
                f'domain covered {s["covered_fraction"] * 100:.1f}%   '
                f'mean depth {s["mean_overlap_depth"]:.2f}',
                legend=(col == 1), channel=channel)

        plt.tight_layout()
        path = save_png(
            out_dir / f'buffer_decomposition_{problem}{suffix}.png', fig=fig)
        plt.close(fig)
        print(f"    saved {path}")

    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trees', type=Path, default=DEFAULT_TREES,
                    help='perfect_trees.json to read the decomposition from')
    ap.add_argument('--problems', nargs='*', default=None,
                    help='subset of problems (default: all in the JSON)')
    ap.add_argument('--quantiles', nargs='*', type=float,
                    default=[0.75, 0.80, 0.85, 0.90],
                    help='quantile levels of the global |du/dz| distribution')
    ap.add_argument('--sigmas', nargs='*', type=float,
                    default=[0.2, 0.1, 0.05, 0.03],
                    help='sigma_fraction values for the current-method '
                         'collar comparison row')
    ap.add_argument('--min-frac', type=float, default=0.0,
                    help='floor on each buffer half-width, as a fraction of '
                         'that side region extent (0 allows zero-width)')
    ap.add_argument('--max-frac', type=float, default=0.5,
                    help='cap on each buffer half-width, as a fraction of '
                         'that side region extent (<=0.5 never reaches a '
                         'second interface)')
    ap.add_argument('--out', type=Path,
                    default=REPO / 'outputs' / 'buffer_decomposition')
    args = ap.parse_args()

    with open(REPO / 'experiments_plan.yaml', 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f).get('base_config', {})
    with open(args.trees, 'r', encoding='utf-8') as f:
        trees = json.load(f)

    problems = args.problems or sorted(trees)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Trees: {args.trees}")
    print(f"Quantiles: {args.quantiles}  min_frac={args.min_frac}  "
          f"max_frac={args.max_frac}")
    print(f"Collar comparison sigma_fraction: {args.sigmas}")
    print(f"Output: {args.out}\n")

    records = {}
    for problem in problems:
        if problem not in trees:
            print(f"  {problem}: not in {args.trees.name}, skipped")
            continue
        try:
            rec = process_problem(problem, trees[problem], base_cfg,
                                  args.quantiles, args.sigmas, args.min_frac,
                                  args.max_frac, args.out)
            if rec is not None:
                records[problem] = rec
        except Exception as exc:                       # noqa: BLE001
            print(f"  ERROR on {problem}: {exc}")
            import traceback
            traceback.print_exc()

    summary = args.out / 'buffer_regions.json'
    with open(summary, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved geometry + stats: {summary}")


if __name__ == '__main__':
    main()
