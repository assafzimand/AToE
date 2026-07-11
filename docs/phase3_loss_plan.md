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

## Loss

```
L =   lambda_r  * L_res( u_theta ; Omega )                    (1) global PDE residual, composition
    + lambda_ic * L_ic ( u_theta )                            (2) exact IC on composition
    + lambda_bc * L_bc ( u_theta )                            (3) exact BC on composition
    + SUM_j ( lambda_ic * L_Gt_j                              (4) u0 guides on the interior
            + lambda_bc * ( L_Gx_j + L_Gx'_j ) )                  faces of Omega_hat_j
                                                                  (skipped if Omega_hat_j empty)
```

- `L_res(v ; S)` = mean squared PDE residual of network `v` over point set `S`.
- `L_Gt_j`  = MSE of `u_j - u0` on the lower-t face of `Omega_hat_j`.
- `L_Gx_j`  = MSE of `u_j - u0` on the two x-faces of `Omega_hat_j`.
- `L_Gx'_j` = MSE of `d/dx u_j - d/dx u0` on the x-faces — only for the periodic set
  (allen_cahn, kdv, ks, schrodinger), i.e. the existing D3 rule, unchanged.

So the method is: **the standard global PINN loss on the composition, plus u0 interface
scaffolding per expert on its exclusive zone's interior faces.**

Why the single global residual term is equivalent to the split
"per-expert on exclusive zones + composed on collars" version: on `Omega_hat_j` the
neighboring windows are identically zero (compact support), so the value and all
derivatives of `u_theta` there involve only `u_j` — same loss value, same gradients.

Implementation notes:
- Keep the forward pass masking experts by window support, so the composed residual on
  10k points does not evaluate every expert on every point (cost then equals the split version).
- Per-expert diagnostics (`[SplitTerms]`-style) can be kept by logging the residual
  restricted to each `Omega_hat_j`.

## What participates where

| Set | Term | Network evaluated | Target / data |
|---|---|---|---|
| interior of `Omega` (uniform points) | PDE residual | `u_theta` (on `Omega_hat_j` this reduces to `u_j` alone; in collars gradients reach all active experts) | — |
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
- **Sampling**: 10k residual points uniform over `Omega`, fed directly to the composed
  residual — no per-expert filtering/allocation at all; redrawn every 500 epochs as now.
  Face guides sampled as now (256 per face), on the `Omega_hat_j` faces.
  (Optional-later knob: densify collar points D6-style in phase 3 if collars look
  under-resolved; not part of this change.)

## Unchanged

- Per-leaf input normalization on `Omega_tilde_j`.
- Soft-composition eval / best-checkpoint selection.
- Root frozen during phase 3.
- Fine-tune stage (pure SSBroyden, D6 optional) — expected to become a short polish.
- Phase-3 budget / optimizer plan (5k Adam + SSBroyden).

## Success criteria (allen_cahn, M=8, same pretrained root)

1. End-of-phase-3 **soft** rel-L2 approaches main's hard-indicator number (~3e-6),
   not just root / 1.67x territory.
2. No blend shock: fine-tune starting rel-L2 ~= end-of-phase-3 rel-L2.
3. Fine-tune changes the metric only marginally (polish, not repair).
