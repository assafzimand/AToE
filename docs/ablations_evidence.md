# Ablations (§5.4) — evidence map

Compiled 2026-09-09 from the saved configs and summaries of all 596 runs in `outputs/experiments/` and the Desktop archive. ★ marks the runs chosen (2026-09-09) as the evidence for each subsection; adaptive-collar-sizing and single-corrector fine-tunes are excluded throughout. Adopted values per `docs/3_method_updated.tex`: overlap fraction σ = 0.03, overlap sampling ratio 0.3, 3 time tiles for KdV / 5 for KS.

Notation: "ft best" = best composed rel-L2 during fine-tuning. On the f32 KdV chains the strong-Wolfe L-BFGS fine-tune froze at epoch 1 in most cells, so "ft best" there equals the epoch-1 PoU blend — i.e. a **pure sweep of the blend at fixed experts**, which is exactly the flat-top-erosion effect the overlap-width ablation is about.

---

## 5.4.1 Overlap width (σ sweep)

### ★ CHOSEN — Allen–Cahn, f64, same experts, same fine-tune (2026-07-13)

`Desktop\allen_cahn_ft_sigma_collar_sweep_20260713_195256\allen_cahn-base-2-60-60-60-60-1-experts-2-20-20-20-1-tanh\` — M8 experts loaded from `roots_checkpoints/allen_cahn_experts.pt` (commit c9c9f36; an older checkpoint than the paper's, 20-wide), 30k SSBroyden fine-tune, collar 0.3:

| σ | subfolder | ft best rel-L2 |
|---|---|---|
| **0.03** | `20260714_102634` | **6.360e-6** |
| 0.05 | `20260714_025846` | 1.072e-5 |
| 0.10 | `20260713_195259` | 1.302e-5 |

Uniform-sampling companions in the same batch: 0.03 → 6.360e-6 (`20260714_141107`), 0.05 → 1.012e-5 (`20260714_063819`), 0.10 → 1.526e-5 (`20260713_232253`) — same ordering. Note: at σ=0.03 the epoch-1 blend is already the best point (the fine-tune drifts upward from 6.36e-6); at 0.05 / 0.10 the blend starts at 2.7e-4 / 9.6e-3 and the fine-tune has to recover.

### ★ CHOSEN (supporting) — KdV f32, adopted higher-order chain: the fine-tune freezes at epoch 1, so these are pure blend-vs-σ sweeps at fixed experts

M18 W3 3x30 on the 50k-Adam root, 2026-08-27, `Desktop\kdv_f32_ft_ablation_lbfgslr_sigma_collar_20260827_145457` (collar 0.3; SW L-BFGS lr 0.1 / 0.01 / 0.001 all identical because the ft freezes at epoch 1):

| σ | ft best (= blend) |
|---|---|
| **0.03** | **1.3746e-2** |
| 0.05 | 1.3741e-2 |
| 0.10 | 2.27e-2 – 2.31e-2 |

Same chain, `Desktop\kdv_f32_ft_ablation2_sw_vs_fs_lr_sigma_2ckpts_20260827_221214`: σ 0.03 → 1.3746e-2 vs σ 0.07 → 1.4318e-2 (17-expert ckpt); 1.3525e-2 vs 1.4226e-2 (11-expert ckpt). Phase-3 partition (hard indicators) 1.3764e-2 — the σ=0.03 blend beats its own partition, σ ≥ 0.07 loses to it.

Paper chain (30k+30k, same A1 experts, phase 3 5.397e-2), widest σ range — `Desktop\kdv_lbfgs_fine_tunes_20260822_171003` (σ 0.05/0.10/0.20/0.30) + `outputs/experiments/kdv_lbfgs_fine_tunes_20260823_050609` (σ 0.03/0.05/0.07):

| σ | ft best (sw lr 1e-3, col 0.3 where available) |
|---|---|
| **0.03** | **5.3760e-2** (the cited paper cell) |
| 0.05 | 5.3924e-2 |
| 0.07 | 5.3931e-2 |
| 0.10 | 5.41e-2 |
| 0.20 | 6.905e-2 |
| 0.30 | 1.144e-1 |

KdV f64 best chain (M12 W2), `outputs/experiments/kdv_f64_ft_ssb_grid_sigma_collar_lr_20260828_123908` (SSBroyden froze; ft best ≈ blend): 0.03 → 1.3127e-5, 0.05 → 1.3206e-5, 0.10 → 2.8031e-5.

### Other supporting (not chosen)

KdV f64 full-domain M20, `Desktop\ks_root_phase3_and_kdv_ft_ablation_20260814_201306` (col 0.3, 40k SSB): σ 0.05 → 8.508e-6, 0.10 → 1.301e-3, 0.20 → 3.641e-3. KdV W1 time-marching, `Desktop\kdv_w12_ftc_sigma_collar_20260812_211320` (col 0.3): 0.05 → 8.55e-7, 0.10 → 2.23e-6, 0.20 → 1.37e-6.

**Bottom line for the text:** 0.03 is best or tied-best everywhere; 0.05 ties on KdV f32/f64 and loses on AC; ≥ 0.10 degrades by 1.6x (KdV f32/f64) to 10x–1000x (AC, KdV full-domain, W1). Mechanism (2026-08-14 laggard-anatomy memo): experts have zero residual points outside their box, so the overlap band is extrapolation — a wide σ erodes the flat tops.

---

## 5.4.2 Overlap sampling ratio in fine-tuning (0.3 vs uniform)

### ★ CHOSEN — uniform vs 0.3 on the PAPER experts, all three PDEs (`outputs/experiments`)

The 2026-07-25 uniform fine-tunes and the 2026-08-05 factorial cd03 cells (= the paper's cited fine-tunes) load byte-identical expert checkpoints (sha256 of `roots_checkpoints/*_experts.pt` at commit f2daad5 == `roots_checkpoints/double_precision/no_resample-4-60-roots/*_experts.pt`, all three PDEs). σ 0.05, fixed collar width (no adaptive sizing), SSBroyden lr 1.0, static 20k pool, tol 1e-10 in both; only `collar_data_ratio` differs (0 vs 0.3). Code is 11 days apart (f2daad5/eb61516 vs 8424579; RAD/SOAP work in between, same fine-tune algorithm).

| PDE | uniform | folder | 0.3 (paper cell) | folder |
|---|---|---|---|---|
| Allen–Cahn | 2.464e-5 (40k) | `finetune_nocorr_40k_20260725_170953/.../20260725_204151` | 2.529e-5 | `ft_noresample_factorial_2x2x2_20260805_063430/.../20260805_081437` |
| Schrödinger | 3.491e-4 (40k) | `finetune_nocorr_40k_20260725_170953/.../20260725_170956` | 3.623e-4 | `.../20260805_095246` |
| Burgers | 1.376e-3 (10k) | `finetune_corrector_ablation_20260725_053553/.../20260725_091154` (the `nocorr` cell = plain joint ft, `single_corrector: false`) | **3.117e-8** | `.../20260805_063435` |

Trajectories (rel-L2 at epoch 1k / 5k / 10k → best): Burgers uniform 3.7e-3 / 1.4e-3 / — → best 1.38e-3 @4k, then rising to 2.5e-3 @9k; Burgers 0.3: 1.5e-4 / 3.3e-6 / 9.4e-7 → 3.1e-8 @32k — the 10k-vs-40k budget mismatch does not explain the gap (400x at 5k, 1000x at 10k). AC uniform vs 0.3 track each other epoch-by-epoch (2.8e-5 vs 2.7e-5 @10k). Schrödinger is root-limited in both arms (root 3.85e-4). The 0.3 cells are also filed under `ac_burgers_schrodinger_experts_creation_20260724_142507-CHECKPOINTS`. Same factorial, ratio 0.5 (fixed σ): AC 2.407e-5, Burgers 2.315e-7, Schrödinger 3.406e-4.

**Reading:** concentrated sampling is neutral on Allen–Cahn and Schrödinger and decisive on Burgers (shock in the overlap band): uniform sampling stalls at 1.4e-3, the 0.3 draw reaches 3e-8.

### Supporting — Allen–Cahn, older experts (2026-07-13, same batch as the σ sweep above)

σ 0.05: uniform 1.012e-5 (`20260714_063819`) vs 0.3 → 1.072e-5 (`20260714_025846`); σ 0.10: 1.526e-5 (`20260713_232253`) vs 1.302e-5 (`20260713_195259`); σ 0.03: identical 6.360e-6 (blend is the best point in both). Neutral on AC, consistent with the chosen pair.

### Supporting — KdV window 1, f64 (2026-08-12)

`Desktop\kdv_w12_ftc_sigma_collar_20260812_211320` — same W1 experts, 40k SSBroyden, only `fine_tune.collar_data_ratio` changes:

| σ | uniform (0) | concentrated (0.3) | factor |
|---|---|---|---|
| 0.05 | 8.55e-7 (ep 1) | 8.55e-7 (ep 1) | — (ft never beat the blend) |
| 0.10 | 5.08e-5 | 2.23e-6 | 23x |
| 0.20 | 3.43e-5 | 1.37e-6 | 25x |

With uniform sampling the fine-tune *degrades* the blend (8.6e-7 → 5.1e-5 at σ=0.10) while concentrated sampling keeps it in the 1e-6 band. KdV f64 full-domain M20 (`Desktop\ks_root_phase3_and_kdv_ft_ablation_20260814_201306`, σ 0.05): uniform 8.628e-6 vs 0.3 → 8.508e-6 (1.4 %).

### Null results (report honestly)

- KdV f32 chains: collar 0 / 0.3 / 0.5 / 0.7 all within 0.1 % (`kdv_lbfgs_fine_tunes_2026082*`, `kdv_AB_root6x128_then_expertsize_20260822_065634`, `kdv_f32_ft_M25global_gbc_sigma003_colgrid_lrgrid_20260908_170917`) — the SW L-BFGS freezes at epoch 1, so sampling never acts.
- KdV f64 M12 grid (2026-08-28): col 0 / 0.3 / 0.5 within 0.01 % (SSB froze).
- KdV W0 f64 (`kdv_w0_ft_ablations2_20260812_081643`): at the measurement floor (~5e-8); uniform 5.48e-8 vs collar 0.5 1.04e-7 — noise-level, not citable either way.

---

## 5.4.3 Time-tile count at fixed M

### ★ CHOSEN — KdV f32, the marked best chain vs the same chain with a single full-domain tree (2026-09-07 vs 2026-09-09)

Verified 2026-09-10: `config_used.yaml` of the two runs differs ONLY in `time_marching.enabled` (true + `only_for_tree_structure` vs false) and in M (25 vs 24, chosen so both decompositions close to the same 24 leaves). Same 3-window root (5.1157e-3), same 3x45+RFF experts (446,667 params in both), same 50k-Adam → SW L-BFGS h100 recipe, global BC, interface order 1 + norm, weights 1e3, 2048-point floor. Code between the two commits (0a82197 → 5039106) only added the hard-IC mount for real time-marching windows, inactive here.

| decomposition | M | leaves | phase-3 best rel-L2 | vs root | stop |
|---|---|---|---|---|---|
| 3 time tiles, global top-M (`kdv_f32_ft_M25global_rootm_lrgrid_then_p3_globalbc_20260907_160942-kdv-best/.../20260907_161222`) | 25 | 24 | **4.899e-3** | 0.96x (below root) | freeze @88,653 |
| single full-domain tree (`kdv_f32_p3_fulldomtree_M24_e3x45_h100_globalbc_20260909_171732-time-tile-ablation/.../20260909_171738`) | 24 | 24 | 5.726e-3 | 1.12x (above root) | freeze @86,047 |

Tiling is worth 1.17x at equal leaf count and capacity, and it is the difference between beating the root and not. Geometry (from `per_expert_rel_l2` bounds): the tiled tree has max box dt = 0.333 and max dx = 0.664; the full-domain tree produces two full-height slabs (dt = 1.0, dx 0.30–0.32) which are its two worst experts (1.47e-2, 1.05e-2) — the "tall-box gamble" predicted by the reference-tree analysis below.

### Older evidence (superseded by the chosen pair)

### Trained fixed-M pair: KdV f64, M = 12, W2 vs W3 (2026-08-25/26)

`outputs/experiments/kdv_p3_trees_from5x60roots_ord1_vs_full_20260825_200910` — same 5x60 root (per order), 3x24 experts, 41k epochs:

| tiles | full interface order | interface order 1 |
|---|---|---|
| W2 linear (11 leaves) | **4.465e-6** | 3.153e-1 (failed: expert 7, wide late leaf) |
| W3 linear_zero | 7.514e-6 | **3.934e-6** |

Mixed: W2 wins at full order, W3 wins at order 1 (and is robust). The marked f64 chain (`kdv_f64_repro_bcnormoff_...20260827_200117`) is W2 (1.314e-5 at its root's 1.313e-5).

### Trained, unwindowed vs W3 at EQUAL leaf count and capacity — f32 200k-Adam line (2026-08-28 → 09-01)

`Desktop\kdv_f32_p3_adam200k_from_200a30l_roots_20260828_212035` (+ nested sub-batches). Same root per row, same 3x30+RFF experts, same 200k flat Adam (no L-BFGS), same 512-point floor, weights 1e3, interface order 1 + norm. The M values differ (15 vs 18) precisely so that the two decompositions have the same number of leaves:

| root (rel-L2) | unwindowed, M15 (16 leaves) | 3 tiles, W3-linear, M18 (16/17 leaves) | factor |
|---|---|---|---|
| with scheduler (1.440e-2) | 8.278e-2 | 3.105e-2 | 2.7x |
| without scheduler (3.132e-2) | 3.675e-2 | 3.174e-2 | 1.16x |

This is the cleanest tile-count comparison in the archive (single knob = tiling, at matched leaf count). Caveats to state: both rows are Adam-only phase 3 (never quasi-Newton-settled) and sit at or above their roots; the "with scheduler" row is the dramatic one but both its cells are 2–6x above the root, the "without" row is the one that belongs to a marked chain (memory: 200k-adam wosched flow).

**Do NOT use** `Desktop\kdv_fulldomain_M20_plain_vs_wtree_20260814_054517` as the windowed arm: its phase 3 OOMed at epoch 2001 (23 leaves = the A10G cliff) and the recorded 0.77 is the untrained state. Its plain partner (`..._20260813_180931`, 18 leaves, 3.42e-5, budget-stopped) is valid but has no surviving windowed counterpart at the same settings; the same-day ifnorm pair (`outputs/experiments/kdv_ifnorm_M18wtree_vs_M12plain_20260814_102953`: plain M12 = 16 leaves 6.744e-6 vs W3-linear M18 = 17 leaves 5.190e-6, both budget-stopped still descending) is the f64 equal-leaf comparison, a 1.3x effect.

### Trained, unwindowed vs W3, different M (not same config)

- f64 full domain, 2026-08-14: unwindowed M12 6.744e-6 vs W3-linear M18 5.190e-6 vs W3-linear M15 5.877e-6 (ifnorm runs; different M).
- f32 200k-Adam line, 2026-08-28/29: unwindowed M15 3.675e-2 (17 leaves, no scheduler) vs W3-linear M18 3.174e-2 vs W3 linear_zero M12 4.01e-2 / 2.01e-2 (3x40 / 3x50). Different M and expert widths.
- f32 full-domain M20 W3-quadratic (23 leaves) OOMed at epoch 1 on A10G (`kdv_fulldomain_M20_plain_vs_wtree_20260814_054517`, 0.77) — the tile count decides leaf count decides memory.

### Reference-tree sweep (no training) — the actual basis for W3

`perfect_tree_examples/sweep/sweep_summary.json` + `perfect_tree_examples/sweep/kdv_no_m_distribution/` (M 10–35, W3, global top-M; scripts `scripts/sweep_perfect_trees*.py`). Leaf counts at fixed M:

| M | unwindowed | W2 | W3 (linear) | W5 (linear) |
|---|---|---|---|---|
| 10 | 15 | — | 13 | 18 |
| 12 | 16 | 11 | 13 | 21 |
| 15 | 16 | 17 | 14 | 23 |
| 18 | 17 | 18 | 17 | 23 |
| 20 | 18 | 19 | 18 | 23 |

W5 costs 5–8 extra leaves at every M (closure overhead of 5 independent trees), W3 ≈ unwindowed. Geometry analyses recorded 2026-08-13 and 2026-09-06 (memory: full-domain experts; f32 flow candidate): raising M 20→30 unwindowed leaves the largest soliton-corridor box unchanged (0.282) and only W5 caps it (0.209); hot-leaf share of root-error mass M20 unwindowed 27 %, W3-quadratic 23 %, W5 21 %; scoring the sweep with the fitted excess-error model (k·area^2.27) put M30-W3-equal as the only top-tier structure, with "W5 = full-width-slab gamble, unwindowed = tall-box gamble". For KdV the 3 tiles also coincide with the 3-window time-marching root (seam-aligned tiles).

KS: 5 tiles = the 5 time-marching windows (dt = 0.1 on T = 0.5, matching Kiyani's 5-window split); the KS reference sweep (`perfect_tree_examples/sweep/ks`, M 10/20/30 × W 5/7/10) shows leaves 39–44 (W5) vs 56–59 (W7) vs 64–79 (W10) at M 20–30. No KS experts run exists.

### Closed

The batch proposed here on 2026-09-09 was run as the ★ pair above (single full-domain tree vs 3 tiles at 24 leaves). A W ∈ {2, 5} extension is optional.

---

## Run index

| purpose | path |
|---|---|
| ★ AC σ sweep (5.4.1) | `Desktop\allen_cahn_ft_sigma_collar_sweep_20260713_195256` |
| ★ KdV f32 σ grids, 50k chain (5.4.1 supporting) | `Desktop\kdv_f32_ft_ablation_lbfgslr_sigma_collar_20260827_145457`, `Desktop\kdv_f32_ft_ablation2_sw_vs_fs_lr_sigma_2ckpts_20260827_221214` |
| KdV f32 σ 0.05–0.30, paper chain | `Desktop\kdv_lbfgs_fine_tunes_20260822_171003`, `outputs/experiments/kdv_lbfgs_fine_tunes_20260823_050609` |
| KdV f64 σ/collar grid | `outputs/experiments/kdv_f64_ft_ssb_grid_sigma_collar_lr_20260828_123908` |
| ★ uniform vs 0.3 on paper experts (5.4.2) | `outputs/experiments/finetune_nocorr_40k_20260725_170953`, `outputs/experiments/finetune_corrector_ablation_20260725_053553` (nocorr cells only), `outputs/experiments/ft_noresample_factorial_2x2x2_20260805_063430` (cd03 cells) |
| KdV W1 uniform vs 0.3 (supporting 5.4.2) | `Desktop\kdv_w12_ftc_sigma_collar_20260812_211320` |
| KdV full-domain σ/collar | `Desktop\ks_root_phase3_and_kdv_ft_ablation_20260814_201306` |
| ★ tiled vs full-domain tree, KdV best chain (5.4.3) | `outputs/experiments/kdv_f32_ft_M25global_rootm_lrgrid_then_p3_globalbc_20260907_160942-kdv-best`, `outputs/experiments/kdv_f32_p3_fulldomtree_M24_e3x45_h100_globalbc_20260909_171732-time-tile-ablation` |
| KdV f64 M12 W2 vs W3 | `outputs/experiments/kdv_p3_trees_from5x60roots_ord1_vs_full_20260825_200910` |
| reference-tree sweeps | `perfect_tree_examples/sweep/` (`sweep_summary.json`, `kdv_no_m_distribution/`, `ks/`, `ks_t05_no_m_distribution/`) |
