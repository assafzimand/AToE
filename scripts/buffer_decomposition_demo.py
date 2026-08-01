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
clipped to the domain box. Mirroring the production collar sizing
(adaptive/collar_sizing.py), each side is then clamped UP to a floor of
``--min-frac`` of the same extent (default 0.03, the run configs'
adaptive_collar_sigma_min): the scan runs from the interface itself, and
a side whose sensor pick lands below the floor is widened to the floor
and drawn with a distinct hatch.

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

In addition to the exploratory multi-panel figures above, every run also
writes a ``paper_images/`` subfolder: the same geometry rendered as many
small, single-panel images (one heatmap, one image per 1-D slice) with no
titles, legends, or outcome shading (no red fill/hatching) -- just the plot
and the tree/collar lines, with every stat (problem, channel, quantile,
slice position) encoded in the filename instead. These are meant to be
picked through and captioned by hand afterwards.

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

    The scan on each side runs from the interface out to
    ``max_frac * span`` of that side's own leaf; the picked half-width is
    then clamped UP to a floor of ``min_frac * span``, exactly as the
    production sizing clamps to sigma_min * extent. Returns
    ``(buffers, refs)`` where each buffer carries its box, whether each
    edge met the threshold (vs. the argmin fallback), and whether it was
    floor-clamped.
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

        edges, hits, floored, edge_g = {}, {}, {}, {}
        for side, leaf, sign in (('lo', itf['low'], -1),
                                 ('hi', itf['high'], +1)):
            span = leaf['bounds_upper'][dim] - leaf['bounds_lower'][dim]
            floor = min_frac * span
            far = z0 + sign * max_frac * span
            far = min(max(far, domain['lower'][dim]), domain['upper'][dim])

            lo_b, hi_b = sorted((z0, far))
            cand = np.nonzero((z_coord >= lo_b - TOL)
                              & (z_coord <= hi_b + TOL))[0]
            if cand.size == 0:                  # grid coarser than the cap
                delta, hits[side] = 0.0, False
            else:
                cand = cand[np.argsort(np.abs(z_coord[cand] - z0))]
                stats = np.array([face_mean(g, int(i), dim, o_mask)
                                  for i in cand])
                below = np.nonzero(stats < ref)[0]
                if below.size:
                    pick, hits[side] = cand[below[0]], True
                else:
                    pick, hits[side] = cand[int(np.argmin(stats))], False
                delta = abs(float(z_coord[pick]) - z0)

            floored[side] = delta < floor - TOL
            edges[side] = z0 + sign * max(delta, floor)
            # Sensor value at the FINAL edge (after clamping): the mean
            # directional gradient the blend transition actually lands on.
            eidx = int(np.argmin(np.abs(z_coord - edges[side])))
            edge_g[side] = face_mean(g, eidx, dim, o_mask)

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
            'floor_clamped_lo': floored['lo'],
            'floor_clamped_hi': floored['hi'],
            'edge_grad_lo': edge_g['lo'],
            'edge_grad_hi': edge_g['hi'],
            'edge_grad_ratio_lo': edge_g['lo'] / ref if ref > 0 else 0.0,
            'edge_grad_ratio_hi': edge_g['hi'] / ref if ref > 0 else 0.0,
            # Percentile of the edge's gradient within the GLOBAL |du/dz|
            # field -- 0.10 means the edge sits in the quietest 10% of the
            # domain, 0.95 means it landed on near-worst-case terrain.
            'edge_grad_rank_lo': float((g < edge_g['lo']).mean()),
            'edge_grad_rank_hi': float((g < edge_g['hi']).mean()),
        })
    return buffers, refs


def region_tiling(leaves, buffers, domain, mode='atomic'):
    """Partition the domain into regions, with buffers counted as regions.

    Cuts the domain on every leaf and buffer coordinate, then labels each
    atomic cell. Cells sharing a label are one region, and the result
    tiles the domain by construction.

    mode='atomic': label by the SET of buffers covering the cell, falling
    back to the containing leaf. Every band overlap (the corner where an
    x-buffer crosses a t-buffer) becomes a region of its own.

    mode='x-first': every collar keeps its whole band as ONE region.
    Overlap over a core belongs to the collar; where collars cross, the
    x-extended collar (interface normal to x) takes the overlap, and the
    t-extended collar keeps only what x-collars left. Same-direction
    collars that overlap are merged into a single region.

    mode='components': every maximal CONNECTED union of collar-covered
    cells is one region, so a transition subdomain is always contiguous —
    crossing and touching collars merge, and nothing is ever split into
    disconnected remnants.

    Returns a dict with the cut coordinates ``xs``/``ts``, the per-cell
    ``labels`` array, the ``is_buf`` mask (cell belongs to a transition
    region), and ``keys`` mapping label id -> ('leaf', i) | ('buf', ids)
    | ('bufx'|'buft', root id) | ('bufc', component id).
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

    is_buf = stack.any(axis=0) if live else np.zeros(CX.shape, dtype=bool)

    # Union-find over live buffers, used by mode='x-first' to merge
    # same-direction collars that overlap.
    parent = list(range(len(live)))

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # Connected components of the collar-covered cells (4-connectivity),
    # used by mode='components'.
    comp = np.full(is_buf.shape, -1, dtype=np.int64)
    if mode == 'components':
        n_comp = 0
        for i0 in range(is_buf.shape[0]):
            for j0 in range(is_buf.shape[1]):
                if not is_buf[i0, j0] or comp[i0, j0] >= 0:
                    continue
                todo = [(i0, j0)]
                comp[i0, j0] = n_comp
                while todo:
                    ci, cj = todo.pop()
                    for ni, nj in ((ci + 1, cj), (ci - 1, cj),
                                   (ci, cj + 1), (ci, cj - 1)):
                        if (0 <= ni < is_buf.shape[0]
                                and 0 <= nj < is_buf.shape[1]
                                and is_buf[ni, nj] and comp[ni, nj] < 0):
                            comp[ni, nj] = n_comp
                            todo.append((ni, nj))
                n_comp += 1

    def _cell_key(i, j):
        if mode == 'components':
            return (('bufc', int(comp[i, j])) if is_buf[i, j]
                    else ('leaf', int(leaf_id[i, j])))
        cov = np.nonzero(stack[:, i, j])[0] if live else ()
        if mode == 'atomic':
            return (('buf', tuple(cov)) if len(cov)
                    else ('leaf', int(leaf_id[i, j])))
        xcov = [k for k in cov if live[k]['dim'] == 0]
        tcov = [k for k in cov if live[k]['dim'] == 1]
        if xcov:
            return ('bufx', _find(xcov[0]))
        if tcov:
            return ('buft', _find(tcov[0]))
        return ('leaf', int(leaf_id[i, j]))

    if mode == 'x-first':
        # First pass: union overlapping same-direction collars. t-collars
        # only union through cells no x-collar claimed, so two t-bands
        # bridged by an x-band stay separate regions.
        for i in range(CX.shape[0]):
            for j in range(CX.shape[1]):
                cov = np.nonzero(stack[:, i, j])[0] if live else ()
                xcov = [k for k in cov if live[k]['dim'] == 0]
                tcov = [k for k in cov if live[k]['dim'] == 1]
                group = xcov if xcov else tcov
                for k in group[1:]:
                    ra, rb = _find(group[0]), _find(k)
                    if ra != rb:
                        parent[rb] = ra

    key_to_id = {}
    labels = np.empty(CX.shape, dtype=np.int64)
    for i in range(CX.shape[0]):
        for j in range(CX.shape[1]):
            labels[i, j] = key_to_id.setdefault(_cell_key(i, j),
                                                len(key_to_id))

    return {'xs': xs, 'ts': ts, 'labels': labels, 'is_buf': is_buf,
            'keys': {v: k for k, v in key_to_id.items()}}


def tiling_segments(tiling):
    """Region-boundary segments of a tiling, split by kind.

    Returns ``(red, black)``: red segments bound a transition region,
    black ones are original leaf interfaces no buffer touched.
    """
    xs, ts = tiling['xs'], tiling['ts']
    labels, is_buf = tiling['labels'], tiling['is_buf']

    red, black = [], []
    for i in range(labels.shape[0] - 1):      # faces normal to x
        for j in range(labels.shape[1]):
            if labels[i, j] != labels[i + 1, j]:
                seg = [(xs[i + 1], ts[j]), (xs[i + 1], ts[j + 1])]
                bucket = red if (is_buf[i, j] or is_buf[i + 1, j]) else black
                bucket.append(seg)
    for i in range(labels.shape[0]):          # faces normal to t
        for j in range(labels.shape[1] - 1):
            if labels[i, j] != labels[i, j + 1]:
                seg = [(xs[i], ts[j + 1]), (xs[i + 1], ts[j + 1])]
                bucket = red if (is_buf[i, j] or is_buf[i, j + 1]) else black
                bucket.append(seg)
    return red, black


def build_region_tiling(leaves, buffers, domain):
    """Back-compat wrapper: ``(red_segments, black_segments, n_regions)``."""
    tiling = region_tiling(leaves, buffers, domain)
    red, black = tiling_segments(tiling)
    return red, black, len(tiling['keys'])


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

    # Each buffer is drawn as its two halves, split at the interface, so
    # every SIDE shows its own outcome: plain red where the edge met the
    # threshold, '///' where it hit the cap and fell back to the argmin,
    # dotted where the sensor pick was clamped UP to the min_frac floor.
    for b in buffers:
        dim = b['dim']
        z0 = b['interface_coord']
        for side in ('lo', 'hi'):
            lo = list(b['bounds_lower'])
            hi = list(b['bounds_upper'])
            if side == 'lo':
                hi[dim] = z0
            else:
                lo[dim] = z0
            w, h = hi[0] - lo[0], hi[1] - lo[1]
            if w <= 0 or h <= 0:
                continue
            if b.get(f'floor_clamped_{side}', False):
                hatch = '...'
            elif not b[f'threshold_met_{side}']:
                hatch = '///'
            else:
                hatch = None
            ax.add_patch(patches.Rectangle(
                (lo[0], lo[1]), w, h, linewidth=0.0, facecolor='#ffffff',
                alpha=0.45, zorder=4))
            ax.add_patch(patches.Rectangle(
                (lo[0], lo[1]), w, h, linewidth=1.0, facecolor='#d62728',
                alpha=0.28, hatch=hatch, edgecolor='#8c0d0d', zorder=5))

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
                          label='collar half — edge met the threshold'),
            patches.Patch(facecolor='#d62728', alpha=0.45, hatch='///',
                          edgecolor='#8c0d0d',
                          label='collar half — hit its cap, fell back to\n'
                                'the local argmin (not set by the sensor)'),
            patches.Patch(facecolor='#d62728', alpha=0.45, hatch='...',
                          edgecolor='#8c0d0d',
                          label='collar half — clamped UP to the floor\n'
                                '(min_frac, i.e. adaptive_collar_sigma_min)'),
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


def draw_clean_panel(ax, leaves, buffers, domain, gt_grid, grid_x, grid_t,
                     channel=None):
    """Ground-truth heatmap with tree/collar geometry, minimally styled.

    Collars are a single semi-transparent red fill -- no hatching, no
    distinction between outcomes (met threshold / argmin / floor). No
    legend, no title. For the paper_images set.
    """
    draw_heatmap(ax, gt_grid, grid_x, grid_t, channel=channel)
    for b in buffers:
        lo, hi = b['bounds_lower'], b['bounds_upper']
        w, h = hi[0] - lo[0], hi[1] - lo[1]
        if w <= 0 or h <= 0:
            continue
        ax.add_patch(patches.Rectangle(
            (lo[0], lo[1]), w, h, linewidth=0.0, facecolor='#ffffff',
            alpha=0.45, zorder=4))
        ax.add_patch(patches.Rectangle(
            (lo[0], lo[1]), w, h, linewidth=1.0, facecolor='#d62728',
            alpha=0.28, edgecolor='#8c0d0d', zorder=5))
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


def single_curve(gt_grid, fixed_dim, idx, channel):
    """One 1-D curve on a grid line: the given channel, or the magnitude
    over channels when ``channel`` is None. Always a single array, so
    paper_images slice plots never need a legend to disambiguate curves.
    """
    sl = gt_grid[idx, :] if fixed_dim == 0 else gt_grid[:, idx]
    if sl.ndim == 1:
        return sl
    return sl[:, channel] if channel is not None else np.linalg.norm(sl, axis=1)


def paper_images(problem, leaves, buffers, domain, gt_grid, grid_x, grid_t,
                 q, n_slices, out_dir, channel=None, channel_suffix=''):
    """Clean, single-panel images for paper use.

    One heatmap (tree outlines black, collar regions filled red) plus one
    image per 1-D slice -- no titles, legends, or outcome shading. Every
    stat that would otherwise be in a title goes into the filename instead.
    Returns the list of saved paths.
    """
    (out_dir / 'paper_images').mkdir(parents=True, exist_ok=True)
    interfaces = find_interfaces(leaves)
    tag = f'{problem}{channel_suffix}_q{q:.2f}'
    paths = []

    t_vals = np.linspace(domain['lower'][1], domain['upper'][1],
                         n_slices + 2)[1:-1]
    x_vals = np.linspace(domain['lower'][0], domain['upper'][0],
                         n_slices + 2)[1:-1]
    jt = [int(np.argmin(np.abs(grid_t - t))) for t in t_vals]
    ix = [int(np.argmin(np.abs(grid_x - x))) for x in x_vals]
    t_vals = [float(grid_t[j]) for j in jt]
    x_vals = [float(grid_x[i]) for i in ix]

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    draw_clean_panel(ax, leaves, buffers, domain, gt_grid, grid_x, grid_t,
                     channel=channel)
    plt.tight_layout()
    paths.append(save_png(out_dir / 'paper_images' / f'{tag}_heatmap.png',
                          fig=fig))
    plt.close(fig)

    for j, t in zip(jt, t_vals):
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.plot(grid_x, single_curve(gt_grid, 1, j, channel),
               lw=1.2, color='black')
        leaf_cuts, buf_cuts = slice_cut_positions(interfaces, buffers, 0, t)
        for c in leaf_cuts:
            ax.axvline(c, color='black', ls='--', lw=1.0)
        for c in buf_cuts:
            ax.axvline(c, color='#d62728', ls=':', lw=1.2)
        ax.set_xlim(float(grid_x[0]), float(grid_x[-1]))
        ax.set_xlabel('x')
        plt.tight_layout()
        paths.append(save_png(
            out_dir / 'paper_images' / f'{tag}_slice_t{t:.3f}.png', fig=fig))
        plt.close(fig)

    for i, x in zip(ix, x_vals):
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.plot(grid_t, single_curve(gt_grid, 0, i, channel),
               lw=1.2, color='black')
        leaf_cuts, buf_cuts = slice_cut_positions(interfaces, buffers, 1, x)
        for c in leaf_cuts:
            ax.axvline(c, color='black', ls='--', lw=1.0)
        for c in buf_cuts:
            ax.axvline(c, color='#d62728', ls=':', lw=1.2)
        ax.set_xlim(float(grid_t[0]), float(grid_t[-1]))
        ax.set_xlabel('t')
        plt.tight_layout()
        paths.append(save_png(
            out_dir / 'paper_images' / f'{tag}_slice_x{x:.3f}.png', fig=fig))
        plt.close(fig)

    return paths


def side_outcome_counts(buffers):
    """(n_floor, n_capped) over buffer SIDES, mutually exclusive.

    A side counts as floor-clamped when the sensor pick was widened to the
    min_frac floor; otherwise as capped when it never met the threshold.
    """
    n_floor = n_capped = 0
    for b in buffers:
        for side in ('lo', 'hi'):
            if b.get(f'floor_clamped_{side}', False):
                n_floor += 1
            elif not b[f'threshold_met_{side}']:
                n_capped += 1
    return n_floor, n_capped


def collar_heatmap_figure(problem, leaves, buffers, domain, gt_grid, grid_x,
                          grid_t, q, out_dir, channel=None,
                          channel_suffix=''):
    """Single large heatmap of the adaptive collars at one quantile, with
    every side's outcome (threshold / cap-argmin / floor-clamped) visible.
    ``channel`` restricts a multi-output problem to a single component.
    """
    n_floor, n_capped = side_outcome_counts(buffers)
    stats = coverage_stats(buffers, grid_x, grid_t)
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    draw_panel(
        ax, leaves, buffers, domain, gt_grid, grid_x, grid_t,
        f'{problem}{channel_suffix} — adaptive collars   |   q = {q:.2f}\n'
        f'{len(buffers)} buffers: {n_capped} sides capped (argmin), '
        f'{n_floor} clamped to the floor   |   '
        f'covered {stats["covered_fraction"] * 100:.1f}%',
        legend=True, channel=channel)
    plt.tight_layout()
    fname = f'collar_heatmap_{problem}{channel_suffix}_q{q:.2f}.png'
    path = save_png(out_dir / fname, fig=fig)
    plt.close(fig)
    return path


def tiling_figure(problem, leaves, buffers, q, domain, gt_grid, grid_x,
                  grid_t, out_dir):
    """Outlines of the re-tiling with CONTIGUOUS transition subdomains.

    Every maximal connected union of collar-covered cells is one
    transition region, so crossing/touching collars merge and no region
    is ever split into disconnected remnants. Only region boundaries are
    drawn: dark red where they bound a transition subdomain, black where
    an original leaf interface survived untouched.
    """
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    draw_heatmap(ax, gt_grid, grid_x, grid_t)

    tiling = region_tiling(leaves, buffers, domain, mode='components')
    red, black = tiling_segments(tiling)
    ax.add_collection(LineCollection(black, colors='black',
                                     linewidths=1.4, zorder=10))
    ax.add_collection(LineCollection(red, colors='#8c0d0d',
                                     linewidths=1.1, zorder=10))

    # The original decision-tree leaves, for reference under the re-tiling.
    for leaf in leaves:
        lo, hi = leaf['bounds_lower'], leaf['bounds_upper']
        ax.add_patch(patches.Rectangle(
            (lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
            linewidth=0.9, linestyle=(0, (4, 3)), edgecolor='black',
            facecolor='none', zorder=9))

    kinds = list(tiling['keys'].values())
    n_core = sum(1 for k in kinds if k[0] == 'leaf')
    n_trans = sum(1 for k in kinds if k[0] == 'bufc')
    ax.set_xlim(domain['lower'][0], domain['upper'][0])
    ax.set_ylim(domain['lower'][1], domain['upper'][1])
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title(f'{problem} — re-tiling, contiguous transitions   |   '
                 f'q = {q:.2f}\n{n_core} cores + {n_trans} transition '
                 f'regions = {n_core + n_trans} subdomains', fontsize=11)
    _legend(ax, [
        Line2D([], [], color='#8c0d0d', lw=1.1,
               label='transition boundary'),
        Line2D([], [], color='black', lw=0.9, linestyle=(0, (4, 3)),
               label='original tree leaf'),
        Line2D([], [], color='black', lw=1.4,
               label='untouched leaf interface'),
    ])

    plt.tight_layout()
    path = save_png(out_dir / f'tiling_{problem}_q{q:.2f}.png', fig=fig)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 1-D slice view
# --------------------------------------------------------------------------

SLICE_COLORS = ('#ff7f0e', '#e377c2', '#17becf', '#bcbd22', '#9467bd')


def slice_cut_positions(interfaces, buffers, dim, other_val):
    """Cut-line positions for a 1-D slice running along dimension ``dim``.

    Returns ``(leaf_cuts, buffer_cuts)``: the tree's subdomain boundaries
    the slice actually crosses (the interface's shared extent must contain
    the fixed coordinate), and the collar (buffer) edges straddling them.
    """
    o = 1 - dim
    leaf_cuts = sorted({itf['z0'] for itf in interfaces
                        if itf['dim'] == dim
                        and itf['o_lo'] - TOL <= other_val
                        and other_val <= itf['o_hi'] + TOL})
    buffer_cuts = []
    for b in buffers:
        if b['dim'] != dim:
            continue
        if (b['bounds_lower'][o] - TOL <= other_val
                <= b['bounds_upper'][o] + TOL):
            buffer_cuts += [b['bounds_lower'][dim], b['bounds_upper'][dim]]
    return leaf_cuts, buffer_cuts


def slice_curves(problem, gt_grid, fixed_dim, idx, channel=None):
    """Curves of the solution on one grid line, as ``(labels, arrays)``.

    ``fixed_dim`` is the dimension held constant at grid index ``idx``;
    the returned arrays run along the other dimension. Multi-output
    problems get one curve per channel plus the magnitude, unless
    ``channel`` selects a single component (then only that curve).
    """
    sl = gt_grid[idx, :] if fixed_dim == 0 else gt_grid[:, idx]
    if sl.ndim == 1:
        return ['u'], [sl]
    names = CHANNEL_NAMES.get(
        problem, tuple(f'c{c}' for c in range(sl.shape[1])))
    if channel is not None:
        label = names[channel] if channel < len(names) else f'c{channel}'
        return [label], [sl[:, channel]]
    labels = [names[c] if c < len(names) else f'c{c}'
              for c in range(sl.shape[1])]
    curves = [sl[:, c] for c in range(sl.shape[1])]
    labels.append('|·|')
    curves.append(np.linalg.norm(sl, axis=1))
    return labels, curves


def slice_view_figure(problem, leaves, buffers, domain, gt_grid, grid_x,
                      grid_t, q, n_slices, out_dir, channel=None,
                      channel_suffix=''):
    """Snapshot figure: heatmaps with marked slice positions on top, then
    ``n_slices`` rows of 1-D slices — u(x) at fixed t on the left, u(t) at
    fixed x on the right. Black dashed lines mark the tree's subdomain
    boundaries crossed by the slice, red dotted lines the collar (buffer)
    edges around them. ``channel`` restricts a multi-output problem to a
    single component, both in the background heatmap and the curves.
    """
    interfaces = find_interfaces(leaves)

    t_vals = np.linspace(domain['lower'][1], domain['upper'][1],
                         n_slices + 2)[1:-1]
    x_vals = np.linspace(domain['lower'][0], domain['upper'][0],
                         n_slices + 2)[1:-1]
    jt = [int(np.argmin(np.abs(grid_t - t))) for t in t_vals]
    ix = [int(np.argmin(np.abs(grid_x - x))) for x in x_vals]
    t_vals = [float(grid_t[j]) for j in jt]
    x_vals = [float(grid_x[i]) for i in ix]

    fig, axes = plt.subplots(n_slices + 1, 2,
                             figsize=(13.0, 2.9 * (n_slices + 1)))

    for col, (title, marks, horizontal) in enumerate((
            (f'{problem}{channel_suffix} — t-snapshots   |   q = {q:.2f}',
             t_vals, True),
            (f'{problem}{channel_suffix} — x-snapshots   |   q = {q:.2f}',
             x_vals, False))):
        ax = axes[0, col]
        draw_panel(ax, leaves, buffers, domain, gt_grid, grid_x, grid_t,
                   title, legend=(col == 0), channel=channel)
        for i, v in enumerate(marks):
            color = SLICE_COLORS[i % len(SLICE_COLORS)]
            line = ax.axhline if horizontal else ax.axvline
            line(v, color='white', lw=2.4, zorder=14)
            line(v, color=color, lw=1.4, ls='--', zorder=15)

    cut_handles = [
        Line2D([], [], color='black', ls='--', lw=1.0,
               label='tree subdomain boundary'),
        Line2D([], [], color='#d62728', ls=':', lw=1.2,
               label='collar (buffer) edge'),
    ]

    for row in range(n_slices):
        for col in range(2):
            ax = axes[row + 1, col]
            if col == 0:                    # u(x) at t = t_row
                coord, cut_dim = grid_x, 0
                fixed_dim, idx, fixed_val = 1, jt[row], t_vals[row]
                xlabel, fixed_name = 'x', 't'
            else:                           # u(t) at x = x_row
                coord, cut_dim = grid_t, 1
                fixed_dim, idx, fixed_val = 0, ix[row], x_vals[row]
                xlabel, fixed_name = 't', 'x'

            labels, curves = slice_curves(problem, gt_grid, fixed_dim, idx,
                                          channel=channel)
            for lab, y in zip(labels, curves):
                ax.plot(coord, y, lw=1.1, label=lab)

            leaf_cuts, buf_cuts = slice_cut_positions(
                interfaces, buffers, cut_dim, fixed_val)
            for c in leaf_cuts:
                ax.axvline(c, color='black', ls='--', lw=1.0)
            for c in buf_cuts:
                ax.axvline(c, color='#d62728', ls=':', lw=1.2)

            color = SLICE_COLORS[row % len(SLICE_COLORS)]
            ax.set_xlim(float(coord[0]), float(coord[-1]))
            ax.set_xlabel(xlabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_title(f'{fixed_name} = {fixed_val:.3f}',
                         fontsize=9, color=color, fontweight='bold')
            for spine in ax.spines.values():
                spine.set_edgecolor(color)

            if row == 0:
                handles = list(cut_handles)
                if len(labels) > 1:
                    handles += ax.get_legend_handles_labels()[0]
                ax.legend(handles=handles, loc='best', fontsize=6.5,
                          framealpha=0.9)

    plt.tight_layout()
    fname = f'buffer_slices_{problem}{channel_suffix}_q{q:.2f}.png'
    path = save_png(out_dir / fname, fig=fig)
    plt.close(fig)
    return path


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
                    max_frac, out_dir, slice_quantiles=(0.9,), n_slices=5,
                    tiling_quantiles=(), paper_n_slices=2):
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
        n_floor, n_capped = side_outcome_counts(buffers)
        _, _, n_regions = build_region_tiling(leaves, buffers, domain)
        sweeps.append((q, buffers, stats, n_capped, n_floor))

        record['sweep'].append({
            'quantile': q,
            'ref_dx': refs[0], 'ref_dt': refs[1],
            'n_buffers': len(buffers),
            'n_capped_sides': n_capped,
            'n_floor_sides': n_floor,
            'n_regions': n_regions,
            'mean_width': float(np.mean(widths)) if widths else 0.0,
            'max_width': float(np.max(widths)) if widths else 0.0,
            **stats,
            'buffers': buffers,
        })
        print(f"    q={q:.2f}  ref=({refs[0]:.4g}, {refs[1]:.4g})  "
              f"{len(buffers)} buffers ({n_capped} sides capped, "
              f"{n_floor} at floor)  "
              f"covered {stats['covered_fraction'] * 100:5.1f}%  "
              f"mean width {np.mean(widths) if widths else 0:.4f}  "
              f"-> {n_regions} regions")
        ranks = [b[f'edge_grad_rank_{s}']
                 for b in buffers for s in ('lo', 'hi')]
        ratios = [b[f'edge_grad_ratio_{s}']
                  for b in buffers for s in ('lo', 'hi')]
        if ranks:
            print(f"           edge |du/dz| percentile in global field: "
                  f"median={np.median(ranks) * 100:4.1f}%  "
                  f"p90={np.quantile(ranks, 0.9) * 100:4.1f}%  "
                  f"max={np.max(ranks) * 100:4.1f}%   |   "
                  f"edge/ref ratio: median={np.median(ratios):.2f}  "
                  f"max={np.max(ratios):.2f}")

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

        for col, (q, buffers, stats, n_capped, n_floor) in enumerate(
                sweeps, start=1):
            draw_panel(
                axes[0, col], leaves, buffers, domain, gt_grid, grid_x, grid_t,
                f'q = {q:.2f}   |   {len(buffers)} buffers, '
                f'{n_capped} sides capped, {n_floor} at floor\n'
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

    for tq in tiling_quantiles:
        match = next((s for s in sweeps if abs(s[0] - tq) < 1e-9), None)
        bufs = match[1] if match is not None else build_buffers(
            leaves, domain, grads, (grid_x, grid_t), tq,
            min_frac, max_frac)[0]
        path = tiling_figure(problem, leaves, bufs, tq, domain, gt_grid,
                             grid_x, grid_t, out_dir)
        print(f"    saved {path}")

    # Standalone collar heatmaps + 1-D slice snapshots, one set per
    # requested quantile, reusing the sweep's buffers when already
    # computed.
    for sq in slice_quantiles:
        match = next((s for s in sweeps if abs(s[0] - sq) < 1e-9), None)
        if match is not None:
            slice_buffers = match[1]
        else:
            slice_buffers, _ = build_buffers(leaves, domain, grads,
                                             (grid_x, grid_t), sq,
                                             min_frac, max_frac)
        print(f"    per-face edge stats at q={sq}:")
        print("      dim  interface  side   width   |du/dz|@edge   /ref  "
              "pctile  outcome")
        for b in slice_buffers:
            for s in ('lo', 'hi'):
                outcome = ('floor' if b[f'floor_clamped_{s}']
                           else 'thresh' if b[f'threshold_met_{s}']
                           else 'argmin')
                print(f"      {DIM_NAMES[b['dim']]:>3}  "
                      f"{b['interface_coord']:+9.3f}  {s:>4}  "
                      f"{b[f'width_{s}']:.4f}  {b[f'edge_grad_{s}']:12.4g}  "
                      f"{b[f'edge_grad_ratio_{s}']:5.2f}  "
                      f"{b[f'edge_grad_rank_{s}'] * 100:5.1f}%  {outcome}")

        path = collar_heatmap_figure(problem, leaves, slice_buffers, domain,
                                     gt_grid, grid_x, grid_t, sq, out_dir)
        print(f"    saved {path}")
        path = slice_view_figure(problem, leaves, slice_buffers, domain,
                                 gt_grid, grid_x, grid_t, sq,
                                 n_slices, out_dir)
        print(f"    saved {path}")

        for channel, suffix, _ in background_variants(problem, gt_grid):
            paths = paper_images(problem, leaves, slice_buffers, domain,
                                 gt_grid, grid_x, grid_t, sq, paper_n_slices,
                                 out_dir, channel=channel,
                                 channel_suffix=suffix)
            print(f"    saved {len(paths)} paper_images "
                  f"({problem}{suffix}, q={sq:.2f})")

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
    ap.add_argument('--min-frac', type=float, default=0.03,
                    help='floor on each buffer half-width, as a fraction of '
                         'that side region extent -- mirrors the production '
                         'adaptive_collar_sigma_min (0 allows zero-width)')
    ap.add_argument('--max-frac', type=float, default=0.5,
                    help='cap on each buffer half-width, as a fraction of '
                         'that side region extent (<=0.5 never reaches a '
                         'second interface)')
    ap.add_argument('--slice-quantiles', nargs='*', type=float,
                    default=[0.9],
                    help='quantiles for the collar heatmap + 1-D slice '
                         'snapshot figures (one set of images per value)')
    ap.add_argument('--n-slices', type=int, default=5,
                    help='snapshots per dimension in the slice figure')
    ap.add_argument('--tiling-quantiles', nargs='*', type=float,
                    default=[0.5, 0.7, 0.9],
                    help='quantiles for the re-tiling figure (transition '
                         'subdomains carved out of the leaves)')
    ap.add_argument('--paper-n-slices', type=int, default=2,
                    help='snapshots per dimension for the clean, '
                         'single-panel paper_images/ output')
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
                                  args.max_frac, args.out,
                                  args.slice_quantiles, args.n_slices,
                                  args.tiling_quantiles, args.paper_n_slices)
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
