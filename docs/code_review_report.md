# AToE Repo Review — clean-run path (root → M-term tree → split experts → PoU fine-tune, + time windows)

Scope: the "cleanest run" configuration (no causal / LRA / FF / RWF / adaptive-sampling / PirateNet). Time-marching verified as requested. File:line references are to the current `main`.

---

## A — Dead / redundant code (leftovers from previous directions)

| # | What | Where | Note |
|---|------|-------|------|
| A1 | `regenerate_training_data()` (~120 lines) | `utils/dataset_gen.py:432` | Never called anywhere (only `resample_residual_inplace` is used). Also imported in 4 trainer modules. |
| A2 | 3-phase bookkeeping: `use_three_phase`, `current_phase`, `phase3_epochs`, `active_cfg`, `_pretrained_force_spawn`, the `epochs = 1` pretrained hack | `trainer/setup.py:825-863`, `trainer/training_context.py:56-60,105` | Computed and stored in ctx but never read — the orchestrator drives phases itself. |
| A3 | `HardIndicator`, `SoftIndicator`, `create_indicator` (~190 lines) | `adaptive/indicators.py:154-342` | Unused; only `BatchedIndicators` is ever instantiated. |
| A4 | Staged-spawning helpers: `get_regions_at_depth`, `get_experts_at_depth`, `get_highest_depth`, `get_union_mask_at_depth`, `count_experts_at_depth`, `get_children_of_parent`, `get_mask_for_expert`, `compute_children_coverage` | `models/atoe_leaves.py:122-221` | No callers. |
| A5 | `spawn_expert`: the `if copy_from_idx ... else ...` branches are byte-identical | `models/atoe_leaves.py:264-277` | Collapse to one branch. |
| A6 | `AToELeaves.__repr__` calls `get_expert_architecture(i)` with an `int` (expects a `RegionDescriptor`) | `models/atoe_leaves.py:562` | Would crash if ever printed with experts — dead + latently broken. |
| A7 | `hasattr(model, 'batched_models')` sync calls | `trainer/setup.py:681`, `trainer/epoch_loop.py:1166` | No model class has `batched_models` (leftover of a removed batched-forward container). The guard always fails; delete. |
| A8 | Copy-pasted import blocks (causal, LRA, plotting, dataset_gen, split helpers) at the top of `setup.py`, `epoch_loop.py`, `finalize.py`, `split_segment.py`, `orchestrator.py` | trainer/* | Most are unused per-module (e.g. `plot_training_curves` in setup, `compute_infinity_norm_error` in split_segment). |
| A9 | `compute_relative_l2_error` | `trainer/utils.py:9` | Unused. |
| A10 | `ctx.rejected_regions`, `ctx.leaf_loss_history` | `trainer/setup.py:1047`, `finalize.py:209-213` | Never written; always empty when passed to `save_regions_metadata`. |
| A11 | Dead config knobs in the clean flow: `sampling.sample_volume_ratio` (its formula is commented out, `dataset_gen.py:45-50`), `wavelet_threshold` / `new_norm_threshold` / `tree_smoothness_threshold` (only read when `AToE_threshold_capacity` is set), `max_experts` (never enforced in `spawn_expert`), legacy `patience_epochs`/`patience_evals`, `_opt_cfg` flat legacy keys, `initial_train_ratio`/`boundary_train_ratio`/eval sizes | config + `trainer/setup.py` | Trim or clearly mark as variant-only. |
| A12 | `_check_output_continuity` / `_log_continuity_diff` | `trainer/orchestrator.py:622-677` | For AToE-Leaves the before/after comparison is base-vs-fresh-random-leaves, so the ">1% change" WARNING fires by construction — pure log noise. |
| A13 | ANT leftovers in `FCNet`: `is_base=False` expert-input path, `get_activation_dim` | `models/fc_model.py:30-36,165-167` | `return_activation` is still used by LS-init; the rest is dead. |
| A14 | Redundant guard `current_optimizer_name != 'LBFGS'` on scheduler step (scheduler only exists on the Adam/SOAP path) | `trainer/epoch_loop.py:494` | Harmless. |
| A15 | Second `compute_wavelet_norms()` call just for diagnostics (recomputes all norms + smoothness with per-node logging) | `trainer/orchestrator.py:448` | Reuse the nodes from `fit_full_tree_and_prune`. |

---

## B — Efficiency: avoidable overhead

> **Status (2026-07-09):** B1, B2, B4, B5, B7, B8, B9 implemented (verified by exact-equivalence tests + smoke run). B3 skipped by decision (joint optimizer kept). B6 skipped — one-time cost at tree build, negligible.

1. **Full-batch Adam goes through a per-row DataLoader.** `batch_size: 99999` ≥ N, so every epoch the `TensorDataset` + custom collate stacks ~10.5k individual GPU rows into 6 tensors (~60k tiny ops/epoch) just to reproduce `train_data`. The LBFGS path already bypasses the loader; do the same for the (effectively full-batch) Adam phase, or slice with a shuffled index tensor. `trainer/setup.py:481-537`, `trainer/epoch_loop.py:348`.

2. **Fine-tune optimizes the retired base.** `_set_trainable(model, 'all')` (`trainer/orchestrator.py:216`) makes the base's ~11.2k params trainable, but the leaves-only forward never touches the base → zero/None gradients forever. With SSBroyden's dense matrix this inflates memory/compute by ≈`((P_base+P_leaves)/P_leaves)²` (~6× for M=8, 20-neuron experts). Fine-tune should use `'leaves'`.

3. **Phase 3 trains all leaves under one joint quasi-Newton optimizer.** The split loss is exactly separable per expert (interface targets are frozen), yet one SSBroyden instance holds a dense `(ΣPᵢ)²` matrix and one shared line search — a hard leaf throttles the step size for every converged leaf. K independent optimizers are mathematically identical, need `Σ Pᵢ²` instead of `(Σ Pᵢ)²` memory, allow per-leaf early stop, and actually deliver the "in parallel" claim of Algorithm 1. (The literature caps dense SSBroyden at ~8×20 nets for exactly this reason.)

4. **Forward evaluates every leaf on every point.** `_forward_soft_only_leaves` runs all K experts on all N inputs although each point lies in the support of ≤ 2^d windows. Masked/gathered evaluation (as in FBPINN) cuts the fine-tune forward + high-order autograd cost by roughly K/2^d. `models/atoe_leaves.py:304-318`.

5. **Split loss does 3–4 tiny forwards per expert per step** (IC / interface_ic / interface_bc / BC), while residuals are already batched into one graph. Batch the face terms the same way. Also `_record` makes ~7·K `.item()` calls per step (GPU syncs). `losses/split_loss.py:235-307,428-442`.

6. **Tree diagnostics do quadratic work.** `_get_parent_id` is an O(node_count) scan per call (`adaptive/region_detector.py:114-125`), `_get_node_bounds` walks it per node, and `_compute_smoothness_indices` + the per-leaf depth-1 "new_norm" tree fits run for **every** node even though the clean config only uses `variable_for_node_accept='norm'`. Gate both on the configured metric.

7. **Checkpoints embed the entire metrics dict** (all per-epoch curves + per-node `spawning_diagnostics`) and are rewritten on every best-improvement. `trainer/setup.py:571-586`. Store metrics only in `metrics.json`.

8. `compute_native_grid_metrics` rebuilds the meshgrid and re-creates numpy→torch tensors on every eval (every 500 epochs, plus twice per segment reconcile). Cache the flattened grid tensors alongside the memoized solution. `trainer/utils.py:82-103`.

9. `BatchedIndicators.update` stores bounds in float32 and re-casts per forward in float64 runs; build them in the default dtype once. `adaptive/indicators.py:409-424,458-462`.

---

## C — Potential bugs / metric issues / paper–pipeline mismatches

> **Status (2026-07-09):** C1–C9 addressed (C1: pairing extended to all periodic problems; C3: KdV only — TM is only used for KdV/KS; C8: log/docstring phrasing separates the M-term budget from expert counts; C9: `eval_blending_mode` flag added per eval epoch — the mixed-regime curve itself is intended). C10–C15 deliberately left open for now.

### Real bugs

1. **Split expert training enforces the wrong BC on periodic problems.** `_add_bc_faces_periodic` marks global x-boundary faces `KIND_BC_TRUE` with `h_gt = 0` (`adaptive/subdomain_data.py:441`), and `_compute_expert_loss` applies Dirichlet matching for every problem except Allen–Cahn (`losses/split_loss.py:291`). KdV, KS, Schrödinger are periodic (their global losses pair left/right + h_x), so during Phase 3 boundary experts are pulled to u=0 at x-boundaries — wrong physics that fine-tune must undo. Fix: extend the cross-expert pairing path to all periodic problems (it's keyed on `is_allen_cahn` only), or mint boundary targets from the frozen root like the interfaces.

2. **Time marching, windows ≥ 1: leaf IC faces use the analytic t=0 IC.** In `_add_ic_face`, a face at the window's `t_min_global` (= window `t_start`) is treated as "true IC" and filled with `_analytic_ic` — the t=0 formula (`adaptive/subdomain_data.py:346-357`). The root phase correctly overrides IC with the previous window's prediction (`trainer/setup.py:43-114`), but the split dataset never does. Experts touching the window start train against the wrong initial data.

3. **KdV (and every non-KS solver) ignores `original_temporal_domain`.** Only `ks_solver` solves the full domain once and serves window slices (`solvers/ks_solver.py:140`). `kdv_solver._get_solution_cached` re-solves the window's `[t_start, t_end]` starting from `cos(πx)` **at t_start** (`solvers/kdv_solver.py:136-188`) — so for a KdV time-marching run all per-window rel-L2 metrics, GT plot backgrounds, and dataset `h_gt` values in windows ≥ 1 come from a wrong reference. Port the KS handling to the other solvers (KdV is the one the paper runs with 4 windows).

4. **TM dataset reuse is fragile.** Every window calls `generate_and_save_datasets(window_cfg)` which writes `datasets/<problem>/training_data.pt` *only if missing* (`utils/dataset_gen.py:118`, `trainer/time_marching.py:517`). If window 0 creates it, it contains only `[0, T/W]` points and windows ≥ 1 filter it to ~nothing. Runs only work when a full-domain dataset happens to pre-exist. Generate per-window files (or always sample the full domain).

5. **Window checkpoints save `adaptive_state: None`.** The method is `state_dict_extended`; the code probes `get_state_dict_extended` (`trainer/time_marching.py:561`), so `window_X_final.pt` can never rebuild the experts/regions.

6. **`pred_after_<segment>` plots are never produced.** `_save_segment_pred_plot` references an undefined `gt_grid` (should be `ctx.gt_grid`), the `NameError` is swallowed by the `try/except`, and every segment logs "prediction plot failed". `trainer/epoch_loop.py:1223`.

### Metric-reporting issues

7. **The TM headline metric contradicts the repo's own convention.** `_compute_full_domain_rel_l2` uses a `RegularGridInterpolator` on an arbitrary 256×200 grid (`trainer/time_marching.py:244-288`), while everywhere else the code insists on native-grid, interpolation-free rel-L2 (its own comment says interpolation inflates errors across fronts). It also takes `pred[:, 0]` only (silently wrong for multi-output). Evaluate `TimeMarchingModel` with `compute_native_grid_metrics` over the original temporal domain instead.

8. **"M-term" vs "M experts".** `M_experts_num` selects the top-M *internal nodes*; the ancestors+siblings closure then determines the leaf set, so the number of spawned experts ≠ M in general (`adaptive/region_detector.py:571-634`). The LaTeX ("budget of M experts", fig. captions "M = 20 experts") conflates the two — either budget on leaf count in code or fix the wording/captions.

9. **Phase-3 metrics are computed under a different composition than the final model.** During split segments eval/best-checkpoint selection uses *hard* indicators (`trainer/split_segment.py:147-152`), so the phase-3 rel-L2 curve and its "best" restore are chosen under hard blending, then fine-tune starts from that point under soft blending. Deliberate, but the single `metrics['rel_l2']` curve silently mixes the two regimes — worth a flag column in metrics and a note in the paper.

### Pipeline ≠ LaTeX (document or align)

10. The LaTeX says fixed full-batch 10k points; the code **resamples collocation points every 500 epochs, including during SSBroyden** (`trainer/epoch_loop.py:244-337`). Justified by the cited literature — but the method section should say so.
11. Interval patience can end segments early and fast-forward Adam→SSBroyden (`epoch_loop.py:1015-1067`); the paper states unconditional 5k + 40k iterations.
12. Interface data are placed only on each leaf's **lower-t face and x-faces**; the upper-t face carries no term (`adaptive/subdomain_data.py:339-459`). Causally sensible, but the LaTeX says "the interior faces" — one clause would fix it.
13. Paper says KdV uses 4 windows; config default is `num_windows: 5`.
14. LaTeX Eq. for Ψ takes the product over `j = 1..d` while the code (correctly, per the text) includes the time dimension — index typo in the paper.
15. `BatchedIndicators` keeps window bounds in float32 even in float64 runs, so region faces sit ~1e-8·scale away from the float64 bounds used to assign split points (`adaptive/indicators.py:409-417`). Cosmetic for soft blending; make it dtype-consistent anyway.

---

## D — Algorithm design: small, literature-backed improvements

1. **Per-leaf input normalization.** Each expert receives raw (x,t) restricted to a small box; deep leaves see tiny coordinate ranges, which conditions a tanh MLP poorly. FBPINN applies "separate input normalisation over each subdomain" (map the leaf box to [-1,1]^d before the net) precisely for this reason — a ~5-line change in `forward_single_expert`/composition with typically large payoff for deep leaves. ([FBPINNs, Moseley et al.](https://arxiv.org/abs/2107.07871), [Springer version](https://link.springer.com/article/10.1007/s10444-023-10065-9))

2. **Train experts on region ∪ collar.** Experts are trained only inside their hard box but the PoU evaluates them up to δ = 0.2·size *outside* it, i.e. exactly where they extrapolate; fine-tune then has to repair the collars. FBPINN-style: sample residual points on the inflated box (bounds ± δ) and put the interface faces at the collar's outer edge. One-line change in `sample_subdomain_residuals` (`adaptive/subdomain_data.py:94-97`).

3. **Richer interface conditions.** The local problems close with value-only Dirichlet data from the root. For 2nd–4th order PDEs this under-constrains near faces (KdV formally needs 3 conditions per x-face); XPINN closes interfaces with value **plus residual continuity** (and optionally normal-flux/derivative terms). You already import `compute_derivatives` in the split loss — adding a first-derivative match to the frozen root on interior faces is cheap. ([XPINN interface conditions](https://ceur-ws.org/Vol-2964/article_60.pdf), [parallel XPINN](https://arxiv.org/pdf/2104.10013))

4. **One Schwarz-style interface refresh.** Expert accuracy at interior faces is floored by root accuracy, since Γ-targets are minted once from u₀. After Phase 3, re-mint interface targets from the (now much better) blended leaves and run a short second expert pass — a single alternating-Schwarz iteration, standard in DD-ML hybrids, and cheaper than relying on global fine-tune to fix interfaces. ([DD-ML survey](https://arxiv.org/pdf/2312.14050))

5. **Per-expert optimizers + per-expert stopping** (same change as B3): with the summed split loss and one shared SSBroyden, one rough leaf keeps all K leaves iterating and shrinks everyone's line-search step. Independent per-leaf optimizers are exact (losses separable), parallel, and let each leaf stop on its own plateau.

6. **Focus fine-tune sampling on collars.** The flat-top interiors are already solved by Phase 3; the global fine-tune's job (per your own §3.6) is reconciling collar zones. Biasing fine-tune collocation toward points where ≥2 windows are active (ψ-overlap mask is already computable from `BatchedIndicators`) spends the 45k-epoch budget where it matters — a sampler-only change.

7. **Down-weight λ_Γ late in expert training.** Γ-targets carry root error; with λ_Γ = 1 the experts are pinned to that error level even when their residual loss is far below the interface MSE. A simple anneal (or dropping the Γ-term once `L_res ≪ L_Γ`) is a small scheme change consistent with penalty-annealing practice in DD-PINN work (e.g. [APINN](https://www.sciencedirect.com/science/article/pii/S0952197623013672)).

8. **Residual-aware M allocation across windows.** `m_distribution: quadratic` hardcodes "later windows are harder". A small data-driven variant — allocate `M_i ∝` the root's per-window residual/error norm after the root pass — matches the adaptivity story and is a ~10-line change in `compute_m_per_window` (`trainer/time_marching.py:37`). (Related idea: PDD segments by residual-loss dynamics — [Luo et al. 2025](https://www.mdpi.com/2227-7390/13/9/1515).)

9. **Optimizer note.** Dense SSBroyden is the right call per [Kiyani et al. 2025](https://arxiv.org/html/2501.16371) / Urbán et al., but its O(P²) matrix caps joint training scale (the papers cap at ~8×20 nets). B2/B3/D5 (leaves-only params + per-leaf optimizers) keep every dense matrix at single-expert size, which is what makes the SSBroyden choice scale with M.

Sources: [FBPINNs](https://arxiv.org/abs/2107.07871) · [FBPINNs (ACM)](https://link.springer.com/article/10.1007/s10444-023-10065-9) · [XPINN generalized space-time DD](https://ceur-ws.org/Vol-2964/article_60.pdf) · [Parallel PINNs via DD](https://arxiv.org/pdf/2104.10013) · [ML+DD survey](https://arxiv.org/pdf/2312.14050) · [APINN](https://www.sciencedirect.com/science/article/pii/S0952197623013672) · [Progressive DD (Luo et al. 2025)](https://www.mdpi.com/2227-7390/13/9/1515) · [Optimizing the Optimizer for PINNs/PIKANs (Kiyani et al. 2025)](https://arxiv.org/html/2501.16371)
