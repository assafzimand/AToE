"""Standalone: hard vs soft (PoU) error heatmaps for a phase-3 checkpoint.

Loads best_model_phase3.pt from a run directory, evaluates the composed model
on the solver's native grid twice — once with HARD indicators (owner-only
readout, what phase 3 trains/reports) and once with the SOFT PoU ψ̃ blend
(what fine-tune/inference use) — and saves a 3-panel figure:

  1. ground truth
  2. log10 |error| with hard indicators + Ω_j tile boundaries
  3. log10 |error| with soft ψ̃      + Ω_j and support Ω̃_j boundaries

Rel-L2 for each mode is reported in the panel titles.

Usage:
    python scripts/plot_phase3_error_maps.py <run_dir> [checkpoint_name] [--sigmas 0.2,0.1,0.05]

    run_dir:         the run output dir (contains config_used.yaml, checkpoints/)
    checkpoint_name: default best_model_phase3.pt
    --sigmas:        comma-separated sigma_fraction values; one figure per value,
                     overriding the collar width of the soft windows at inference
                     (default: the trained model's own sigma_fraction).
"""

import sys
import math
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.atoe_leaves import AToELeaves


def load_model(run_dir: Path, ckpt_name: str, device):
    cfg = yaml.safe_load(open(run_dir / "config_used.yaml", encoding="utf-8"))
    if cfg.get("precision", "float32") == "float64":
        torch.set_default_dtype(torch.float64)

    ckpt_path = run_dir / "checkpoints" / ckpt_name
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "adaptive_state" not in ckpt:
        raise ValueError(f"{ckpt_path} has no adaptive_state (not an AToE checkpoint)")

    model = AToELeaves(
        base_architecture=cfg["base_architecture"],
        activation=cfg["activation"],
        config=cfg,
        adaptive_config=cfg["adaptive_pinn"],
        experts_architecture=cfg.get("experts_architecture"),
    )
    model.load_state_dict_extended(ckpt["adaptive_state"])
    model = model.to(device)
    model.eval()
    print(f"Loaded {ckpt_path.name}: {model.num_experts} experts, "
          f"leaves={sorted(model.leaf_indices)}, "
          f"ckpt rel_l2={ckpt.get('rel_l2')}, epoch={ckpt.get('epoch')}")
    return model, cfg


def native_grid(cfg):
    import importlib
    problem = cfg["problem"]
    solver = importlib.import_module(f"solvers.{problem}_solver")
    x_grid, t_grid, h_sol = solver._get_solution_cached(cfg)
    t0, t1 = cfg[problem]["temporal_domain"]
    t_mask = (t_grid >= t0 - 1e-12) & (t_grid <= t1 + 1e-12)
    return np.asarray(x_grid), np.asarray(t_grid)[t_mask], np.asarray(h_sol)[t_mask]


def predict(model, xt, device, chunk=65536):
    dtype = next(model.parameters()).dtype
    out = []
    with torch.no_grad():
        for s in range(0, xt.shape[0], chunk):
            xb = torch.tensor(xt[s:s + chunk], dtype=dtype, device=device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def rel_l2(pred, gt):
    return math.sqrt(((pred - gt) ** 2).sum()) / (math.sqrt((gt ** 2).sum()) + 1e-10)


def leaf_boxes(model):
    """(lower, upper) arrays (K, 2) for leaf tiles, plus expanded supports."""
    leaf = [i for i in sorted(model.leaf_indices) if i >= 0]
    lower = np.array([model.regions[i].bounds_lower for i in leaf])
    upper = np.array([model.regions[i].bounds_upper for i in leaf])
    delta = np.maximum(model.sigma_fraction * (upper - lower), 1e-6)
    return lower, upper, lower - delta, upper + delta


def draw_boxes(ax, lower, upper, **kw):
    from matplotlib.patches import Rectangle
    for lo, hi in zip(lower, upper):
        ax.add_patch(Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
                               fill=False, **kw))


def make_figure(model, cfg, device, run_dir, ckpt_name, sigma, grid_data):
    """One 3-panel figure (GT / hard error / soft error) for a given sigma."""
    x_grid, t_grid, h_sol, xt, gt = grid_data

    # Override the collar width of the soft windows at inference
    model.sigma_fraction = float(sigma)
    model.sync_batched_indicators()

    results = {}
    orig_mode = model.blending_mode
    for mode in ("hard", "soft"):
        model.blending_mode = mode
        pred = predict(model, xt, device)
        err = np.abs(pred - gt).max(axis=1).reshape(h_sol.shape)  # (nt, nx)
        results[mode] = (rel_l2(pred, gt), err)
        print(f"sigma={sigma:<5g} {mode:5s}: rel-L2 = {results[mode][0]:.6e}, "
              f"max|err| = {err.max():.4e}")
    model.blending_mode = orig_mode

    lower, upper, lo_exp, hi_exp = leaf_boxes(model)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = [x_grid.min(), x_grid.max(), t_grid.min(), t_grid.max()]
    gt_img = np.abs(h_sol) if np.iscomplexobj(h_sol) else h_sol

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    im0 = axes[0].imshow(gt_img, origin="lower", aspect="auto", extent=extent,
                         cmap="RdBu_r")
    axes[0].set_title("Ground truth")
    plt.colorbar(im0, ax=axes[0])

    # Log-scale |err| panels, shared color scale across both
    log_hard = np.log10(results["hard"][1] + 1e-16)
    log_soft = np.log10(results["soft"][1] + 1e-16)
    vmin = min(log_hard.min(), log_soft.min())
    vmax = max(log_hard.max(), log_soft.max())

    im1 = axes[1].imshow(log_hard, origin="lower", aspect="auto", extent=extent,
                         cmap="hot", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"HARD indicators | rel-L2 = {results['hard'][0]:.3e} "
                      f"| max = {results['hard'][1].max():.2e}")
    draw_boxes(axes[1], lower, upper, edgecolor="lime", linewidth=1.0)
    plt.colorbar(im1, ax=axes[1], label="log10 |err|")

    im2 = axes[2].imshow(log_soft, origin="lower", aspect="auto", extent=extent,
                         cmap="hot", vmin=vmin, vmax=vmax)
    axes[2].set_title(f"SOFT PoU ψ̃ (σ={sigma:g}) | "
                      f"rel-L2 = {results['soft'][0]:.3e} "
                      f"| max = {results['soft'][1].max():.2e}")
    draw_boxes(axes[2], lower, upper, edgecolor="lime", linewidth=1.0)
    draw_boxes(axes[2], lo_exp, hi_exp, edgecolor="cyan", linewidth=0.9,
               linestyle="--")
    plt.colorbar(im2, ax=axes[2], label="log10 |err|")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("t")

    plt.tight_layout()
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    sig_tag = f"_sigma{str(sigma).replace('.', 'p')}"
    out_path = out_dir / f"error_maps_{ckpt_name.replace('.pt', '')}{sig_tag}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return results["soft"][0]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sigmas = None
    for a in sys.argv[1:]:
        if a.startswith("--sigmas"):
            sigmas = [float(s) for s in a.split("=", 1)[-1].split(",")]
    run_dir = Path(args[0]) if args else None
    ckpt_name = args[1] if len(args) > 1 else "best_model_phase3.pt"
    if run_dir is None:
        raise SystemExit("usage: plot_phase3_error_maps.py <run_dir> "
                         "[checkpoint] [--sigmas=0.2,0.1,0.05]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(run_dir, ckpt_name, device)
    if sigmas is None:
        sigmas = [model.sigma_fraction]

    x_grid, t_grid, h_sol = native_grid(cfg)
    T, X = np.meshgrid(t_grid, x_grid, indexing="ij")           # (nt, nx)
    xt = np.column_stack([X.ravel(), T.ravel()])
    gt = h_sol.reshape(-1, 1) if not np.iscomplexobj(h_sol) else \
        np.column_stack([h_sol.real.ravel(), h_sol.imag.ravel()])
    grid_data = (x_grid, t_grid, h_sol, xt, gt)

    summary = {}
    for sigma in sigmas:
        summary[sigma] = make_figure(model, cfg, device, run_dir,
                                     ckpt_name, sigma, grid_data)

    print("\nSoft rel-L2 by sigma_fraction:")
    for s, r in summary.items():
        print(f"  sigma={s:<6g} rel-L2 = {r:.6e}")


if __name__ == "__main__":
    main()
