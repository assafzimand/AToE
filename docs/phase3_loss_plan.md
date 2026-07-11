# Plan: Phase-3 loss redesign ("exact-object decomposition")

## Geometry (per leaf j, all axis-aligned)

- `Omega_j` — hard leaf region (disjoint, they tile the domain `Omega`).
- `Omega_tilde_j = (Omega_j +- delta_j) ∩ Omega` — window support (unchanged; still the per-leaf normalization box).
- **Exclusive zone** `Omega_hat_j = { x : normalized weight of expert j == 1 }`
  = `Omega_j` minus the ramps of neighboring windows that reach into it.
  No shrink on faces lying on the physical boundary or `t=0` (nothing penetrates there).
  On this set the composition equals the expert exactly: `u_theta == u_j`.
- **Collar set** `C = { x in Omega : >= 2 windows active }`.
- These partition the domain (no overlap, no gaps):

  `Omega = (union over j of Omega_hat_j) ∪ C`

- **Swallowed leaf**: `Omega_hat_j` is empty (neighbor collars cover all of `Omega_j`).
  Must be detected and handled, not crash.

## Phase-3 loss (FINAL form — grouped exact-object)

```
L_phase3 =
      lambda_r  * ( SUM_j L_res( u_theta ; Omega_hat_j )      (1) solo-zone residual means
                  +       L_res( u_theta ; C ) )                  + one collar mean
    + lambda_ic * L_ic ( u_theta )                            (2) exact IC on composition
    + lambda_bc * L_bc ( u_theta )                            (3) exact BC on composition
    + SUM_j s(e) * ( lambda_ic * L_Gt_j                       (4) u0 guides on the interior
                   + lambda_bc * ( L_Gx_j + L_Gx'_j ) )           faces of Omega_hat_j
                                                                  (skipped if Omega_hat_j empty)
```

- `L_res(v ; S)` = mean squared PDE residual of network `v` over the points in set `S`.
- `L_Gt_j`  = MSE of `u_j - u0` on the lower-t face of `Omega_hat_j`.
- `L_Gx_j`  = MSE of `u_j - u0` on the two x-faces of `Omega_hat_j`.
- `L_Gx'_j` = MSE of `d/dx u_j - d/dx u0` on the x-faces — only for the periodic set
  (allen_cahn, kdv, ks, schrodinger), i.e. the existing D3 rule, unchanged.
- `s(e)` = optional linear interface anneal (D7), default off (`s == 1`).

Every physics term is evaluated on `u_theta`, the reported object. On `Omega_hat_j`
the neighboring windows are identically zero (compact support), so
`L_res(u_theta; Omega_hat_j)` IS the expert's well-posed local residual — same value,
same gradients as evaluating `u_j` alone.

**Why a sum of per-group means and not one global mean** (N1_clean, 2026-07-11):
with a single global mean the loss converged to 2e-13 while rel-L2 froze at 3.50e-5 —
worse than the root. Each point carried weight 1/10000, so the tree's small
high-detail zones (the sharp features it was built to isolate) contributed ~1% of the
loss while showing the LARGEST per-zone residuals all run long. One weight-1 mean per
group restores the per-region weighting the pre-redesign per-expert loss had
(per-point weight `1/n_group`, i.e. ~25-100x more for the small zones).

Grouping tag = the `expert_id` column of the residual rows: solo owner via
exclusive-box membership, `-1` for collar points. Swallowed leaves own no solo group.
An expert whose solo group happens to be empty on a draw simply has no mean until the
next resample (logged as a warning).

## Fine-tune loss

Classic PINN on the composition — nothing else:

```
L_ft =  lambda_r  * L_res( u_theta ; Omega )                  uniform points, global mean
      + lambda_ic * L_ic ( u_theta )
      + lambda_bc * L_bc ( u_theta )
```

No guides, no grouping, no collar sampling; pure SSBroyden (5k), leaf experts only.
Under the exact-object phase 3 this stage is a polish: N1 showed its best epoch was
its FIRST — phase 3 already leaves the composition at the loss's minimum.

Implementation notes:
- Keep the forward pass masking experts by window support, so the composed residual on
  10k points does not evaluate every expert on every point.
- Per-expert diagnostics (`[SplitTerms]`-style): each expert's recorded residual is
  its solo-group mean; the collar mean is logged on the composition line.

## What participates where

| Set | Term | Network evaluated | Target / data |
|---|---|---|---|
| `Omega_hat_j` (solo points, one weight-1 mean per expert) | PDE residual | `u_theta` == `u_j` there | — |
| `C` (collar points, one weight-1 mean) | PDE residual | `u_theta` (gradients reach all active experts) | — |
| interior faces of `Omega_hat_j` (not on `t=0` / physical boundary) | guide | `u_j` | frozen `u0`: values on lower-t face; values + x-derivative on x-faces for periodic problems |
| `t=0` | IC | `u_theta` | exact IC, 256 rows, unchanged |
| spatial boundary | BC | `u_theta` | exact BC, 256 rows, unchanged |

## What is removed vs. current AToE branch

1. Per-expert residual on the **inflated** box — replaced by the single composed residual (1). *(A1)*
2. `u0` guides on faces lying on `t=0` / physical spatial boundary — gone; exact data on
   `u_theta` is the only supervision there. *(A2)*
   **Code note:** `adaptive/subdomain_data.py` currently guides ALL outer faces by design
   (its docstring says so); `_add_t_interface_face` / `_add_x_interface_faces` already
   compute `on_t_min` / `_is_global_boundary` — flip these from "include anyway" to
   "skip face entirely".
3. Guide faces move from the outer `Omega_tilde_j` faces **inward** to the `Omega_hat_j`
   faces (and per note 2, a face of `Omega_hat_j` that coincides with `t=0` or the
   physical boundary gets no guide either — only the composition IC/BC covers it).

## Special cases & knobs

- **Swallowed leaves**: no guide terms — trained purely through the global composed
  residual (1), FBPINN-style. Log which leaves this applies to and monitor their
  training curves; a fading guide (D7 with w=1) is the deferred fallback.
- **D7 anneal** on the guide terms (4): keep as an optional knob, default off (w=0).
- **Sampling**: 10k residual points uniform over `Omega`, fed to the grouped composed
  residual — no per-expert filtering/duplication; redrawn every 500 epochs as now.
  Face guides sampled as now (256 per face), on the `Omega_hat_j` faces.
  (Optional-later knob: densify collar points D6-style in phase 3 if collars look
  under-resolved; not part of this change.)

## Unchanged

- Per-leaf input normalization on `Omega_tilde_j`.
- Soft-composition eval / best-checkpoint selection.
- Root frozen during phase 3.
- Fine-tune: 5k pure SSBroyden on the classic composed PINN loss (see above).
- Phase-3 budget / optimizer plan: 35k = 5k Adam + 30k SSBroyden.

## Success criteria (allen_cahn, M=8, same pretrained root)

1. End-of-phase-3 **soft** rel-L2 approaches main's hard-indicator number (~3e-6),
   not just root / 1.67x territory.
2. No blend shock: fine-tune starting rel-L2 ~= end-of-phase-3 rel-L2.
3. Fine-tune changes the metric only marginally (polish, not repair).
