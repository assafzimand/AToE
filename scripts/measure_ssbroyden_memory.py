"""Measure SSBroyden peak GPU memory versus parameter count and find the
largest model the card can optimize with it.

Runs a synthetic MLP regression (the optimizer's memory depends only on the
parameter count n, not on the PDE), takes 3 SSBroyden steps per size, records
torch.cuda.max_memory_allocated() above the baseline, fits peak = k * n^2 * 8 B
and reports the n at which the card's budget is reached, plus the equivalent
root architectures. Stops at the first OOM.

Usage (on the GPU machine, ~1-2 minutes):
    python scripts/measure_ssbroyden_memory.py [--budget-gib 20] [--f32]
"""
import argparse, math, gc, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--budget-gib', type=float, default=20.0, help='usable GPU budget to solve for')
ap.add_argument('--f32', action='store_true', help='measure in float32 instead of float64')
ap.add_argument('--max-width', type=int, default=200)
args = ap.parse_args()

assert torch.cuda.is_available(), 'needs a CUDA device'
dev = torch.device('cuda')
dtype = torch.float32 if args.f32 else torch.float64
torch.set_default_dtype(dtype)
bytes_per = 4 if args.f32 else 8
from scimba_torch.optimizers.ssbroyden import SSBroyden

print(f"device {torch.cuda.get_device_name(0)} | total {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB | dtype {dtype}")
print(f"{'width':>5} {'n':>7} {'Hk GiB':>8} {'peak GiB':>9} {'peak/Hk':>8} {'s/step':>7}")
rows = []
# 2-hidden-layer MLP [2, w, w, 1]: n = w^2 + 5w + 1
for width in range(20, args.max_width + 1, 10):
    net = torch.nn.Sequential(torch.nn.Linear(2, width), torch.nn.Tanh(), torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, 1)).to(dev)
    n = sum(p.numel() for p in net.parameters()); hk = n * n * bytes_per / 2**30
    X = torch.rand(2000, 2, device=dev); Y = torch.sin(3 * X.sum(1, keepdim=True))
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    try:
        opt = SSBroyden(list(net.parameters()), lr=1.0, tolerance_grad=1e-12, method='ssbroyden')
        def closure():
            opt.zero_grad(); l = ((net(X) - Y) ** 2).mean(); l.backward(); return l
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(3):
            opt.step(closure)
        torch.cuda.synchronize(); dt = (time.time() - t0) / 3
        peak = (torch.cuda.max_memory_allocated() - base) / 2**30
        print(f"{width:>5} {n:>7d} {hk:>8.3f} {peak:>9.3f} {peak / hk:>8.2f} {dt:>7.2f}")
        rows.append((n, peak))
        del opt
    except torch.cuda.OutOfMemoryError:
        print(f"{width:>5} {n:>7d} {hk:>8.3f}  OOM (reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB)")
        del net; gc.collect(); torch.cuda.empty_cache()
        break
    del net, X, Y; gc.collect(); torch.cuda.empty_cache()

if len(rows) >= 3:
    # fit peak = k * n^2 * bytes (through the origin on the large-n points)
    big = [(n, p) for n, p in rows if n >= rows[-1][0] / 3]
    k = sum(p * n * n for n, p in big) / sum((n * n) ** 2 for n, p in big) / (bytes_per / 2**30)
    print(f"\nfit on the {len(big)} largest sizes: peak ~ {k:.2f} x n^2 x {bytes_per} B")
    n_max = math.sqrt(args.budget_gib * 2**30 / (k * bytes_per))
    w4 = (-7 + math.sqrt(49 - 12 * (1 - n_max))) / 6
    print(f"budget {args.budget_gib:.0f} GiB -> n_max ~ {n_max:,.0f} params "
          f"(~ 4x{w4:.0f} plain MLP root; 4x60 = 11,221; 5x60 = 14,881; "
          f"6x128+RFF window root = 99,201; KdV 24 experts 3x45+RFF = 149,064)")
