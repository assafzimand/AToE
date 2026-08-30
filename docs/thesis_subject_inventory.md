# Thesis subject inventory — everything examined across NCC-PINN and AToE

Compiled 2026-08-30 from four evidence layers:

1. **NCC-PINN repo** — 1,176 commits across 11 branches (Nov 2025 → Jul 2026), docs/, .ai/PRD.md, experiment plans.
2. **AToE repo** — 250 commits (2026-07-04 → 08-30), branches `main`, `AToE`, `AToE-Schwarz`, `experts24-oldcommit`, `lora_tests`; docs/3_method_updated.tex; ~2,850-line experiments_plan.yaml.
3. **Chat transcripts** — 21 Claude Code sessions (Jul 27 → Aug 30, 2026), 1,161 user messages mined.
4. **Persistent memory notes** — 15 distilled result/decision files.

**Excluded per your instruction:** the NCC-PINN inner-behaviour era (Nov 2025 → ~Jan 21 2026): NCC = Nearest Class Centroid representation analysis, linear probes, frequency tracker, derivatives tracker, single-network capacity sweeps, and the `residual_norm_control` (RNC) branch. One bridge fact kept: those capacity sweeps later fed tree-era expert sizing (architecture bank / capacity maps).

---

## 1. Lineage timeline (one line per era)

| When | Where | What |
|---|---|---|
| Jan 21 – Feb 6, 2026 | NCC-PINN `geo-wavelet-domain-decomp` | First tree decomposition: RF geometric wavelets, spawn-by-depth, hard vs soft indicators, pretrained base + caching |
| Feb 7 – Feb 26 | NCC-PINN `ANT` / `AToE` / `AToE-New` / `AToE-Final` | Birth of the three composition variants; spawning rules; norm-threshold searches; ANT branch abandoned Feb 19 |
| Mar 19 – Apr 30 | NCC-PINN `AToE-AToE-leaves-ANT` | M-term trees, perfect (GT-oracle) trees, 150-expert stress test, KdV/KS added, "7 SOTA PINN features", smoothness-derived trees |
| Apr 22 – Jun 18 | same | 39-run "splits on the go" KS campaign; `M_term_tree_by_norm` becomes the sole method; time marching for KS; Kiyani/SSBroyden reproduction |
| Jun 21 – Jun 28 | NCC-PINN `M_term_AToE-AToE-leaves-ANT` | Compact smoothstep windows, additive composition, staged orchestrator, split-icbc losses; PirateNet KdV reproduction |
| Jun 28 – Jul 6 | NCC-PINN `Leaves-layer-only` | PDD-style per-leaf split training vs additive; root-checkpoint workflow; native-grid eval; handoff to the AToE repo |
| Jul 4 – Jul 9 | AToE `main` | Import + cleanup; paper-grade GTs (ETDRK4/Cole-Hopf); native-grid metrics; symmetry-aware tree fitting |
| Jul 9 – Jul 13 | AToE branch `AToE` | Phase-3 loss redesigns: exact-object composed loss, u0-guided interfaces, **owner-imitator** phase 3 (abandoned) |
| Jul 12 | AToE branch `AToE-Schwarz` | **Schwarz-scheduled** phase 3: distill warm start + freeze/unfreeze blocks (abandoned) |
| Jul 13 – Jul 24 | AToE `main` | Interface-trained-experts pipeline consolidation; collar data sampling; corrector ft; paper experts creation (24w 2nd-order run) |
| Jul 25 – Aug 5 | AToE `main` | SOAP/f32 line, GPU profiling, adaptive collar sizing, RAD benchmarks, fine-tune recipe search |
| Aug 7 – Aug 16 | AToE `main` | KdV/KS higher-order campaign: window pathologies, interface normalization breakthrough, full-domain experts, wtree |
| Aug 16 – Aug 23 | AToE `main` | Adam→LBFGS f32 roots, LBFGS freeze forensics, interface/BC order × normalization ablations |
| Aug 23 – Aug 24 | AToE `main` + `experts24-oldcommit` | Paper 24w reproduction, sigma×collar ft grids on the 2nd-order trio |
| Aug 24 – Aug 30 | AToE `main` + `lora_tests` | Dual precision lines (f64 SSBroyden vs f32 LBFGS), BC-norm retirement, LoRA stages, "marked" citable result chains |

---

## 2. Compositions and tree-model designs

- **AToE (additive)** — experts as additive corrections over *all* tree nodes, per-level PoU with base term w = Ψ/(1+ΣΨ) (FBPINN-style background normalization); trimmed-tree semantics; sibling-retention-drop proposal.
- **ANT (Adaptive Neural Tree)** — parent hidden-activation propagated to children, leaves-only PoU; branch abandoned Feb 19, 2026 with no written negative-result rationale.
- **AToE-Leaves (non-additive, leaves-only PoU, root retires)** — the adopted formulation, carried into the AToE repo.
- Framework name in NCC-PINN docs: **PAFA** ("Physics-informed Adaptive Framework with Adaptive domain decomposition", .ai/PRD.md + draft paper PDF Jun 21).
- Collapse-to-vanilla diagnosis: all-nodes PoU makes experts compete with the root — the structural motivation for the additive variant.
- No learned router/gating anywhere — windows are always fixed geometric PoU; a trainable APINN-style gate was only *discussed* (fine_tune_options_plan.md, Method D), never implemented.
- Hard vs soft indicator functions; sigmoid-product windows (C∞, non-compact) vs **compact smoothstep flat-top windows** S_N with C^N ≥ PDE order.
- "Transition-region tiling" MoE idea — shrink leaves in favor of dedicated transition subdomains built from adaptive collars (discussed Aug 1, not implemented).
- Corrector-network composition (additive corrector over the blend); later **single_corrector** ft (χ = 1−Σψ̃², zero in flat-tops, 921 trainable params).
- Vanilla PINN root as the reported baseline; LoRA adapters for root refine + fine-tune (`lora_tests` branch).

## 3. Tree construction and splitting criteria

- RF geometric wavelets: ψ = Q_child − Q_parent, score = ‖ψ‖²·volume — the core splitting signal.
- Criterion variants each tried: residual-weighted wavelet norms; loss-weighted wavelet norms; "new_norm" (sum of children's classic norms); **Tree-Besov smoothness index** α (slope of log‖ψ‖/|ν|^½ vs log|ν| over descendants, R²-gated); volume- vs n_samples-normalization.
- Acceptance policies: per-problem thresholds (later removed) → **M-term construction** (full wavelet tree from converged root, top-M coefficients + ancestor/sibling closure to a complete binary tree) as the sole method.
- Spawning policies (incremental era): spawn-by-depth schedule; one-child vs both-children; loss-based spawning (split worst leaf); plateau gating with retries/cooldown; **"splits on the go"** (incremental splitting during training, 39-run KS campaign) — demoted to future work.
- **Perfect trees**: oracle decompositions built from ground truth for each criterion; runtime `use_perfect_trees` mode; sweep scripts M ∈ {6…30} × windows W ∈ {2,3,5} × allocation {equal, linear, quadratic, linear_zero, quadratic_zero}.
- Tree fitted on the root *prediction*, not the error (deliberate); native-solver-grid vs 200×200 lattice fit drift between commits (changes split candidates); symmetry-aware fitting with epsilon tie acceptance; explicit domain box; `tree_min_samples_leaf` semantics.
- Time tiling: independent tree per time window, budget M allocated uniform/linear/quadratic(+`_zero` variants) across windows; **wtree** = windows shape the tree only (`only_for_tree_structure`), training stays full-domain.
- Diagnostics used for design: capacity maps (params/volume), corridor-volume analysis, hot-leaf share of root-error mass, wavelet-norm-vs-worst-expert correlation.

## 4. Expert training schemes (phase 3) — the "how do experts learn their region" axis

- Experts on **full global loss** vs on **owned/shrunk subdomains** — both flows existed in NCC-PINN (variant_training_flow.md).
- **PDD-style per-leaf split training** (the adopted scheme): residual inside Ω_j + interface matching to the frozen root u0 on interior faces + true IC/BC on physical boundary + neighbor continuity + periodic-BC pairing across edge experts.
- Additive split variant: leaves trained as corrections on a frozen-but-differentiable root, zero interface targets.
- **Owner-imitator phase 3** (AToE branch, New_phase_3.tex): all loss terms on expert outputs, composition as readout — abandoned.
- **Schwarz-scheduled phase 3** (AToE-Schwarz branch): distill warm start, colored freeze/unfreeze blocks, group-normalized block residual, SSBroyden distill stage; test runs on Allen-Cahn/Burgers — abandoned.
- Exact-object phase-3 loss: composed residual + guides restricted to exclusive-zone faces; grouped per-solo-zone means.
- Coarse-to-fine staged level-by-level training with per-level hard freeze; freeze_mode none/previous/base_only + ancestor-freeze with grouped optimizer — all later removed.
- Full-domain experts (no marching, no window restriction); zero-experts-in-window mechanism.
- Expert init at spawn: parent-weight copy + zeroed output (additive) vs copied output (Leaves/ANT continuity); glorot; LS output init; spectral-norm wrapping; init-damage measurement (arch mismatch → only output layer copied → Adam "demolition" phase); distill-init proposed, explicitly rejected by you.
- Per-expert machinery: min_points floors (512/1024/2048/4096), volume-proportional budgets, per-expert gradient clipping, per-leaf causal state (removed), per-leaf adaptive sampling.
- Experts-only runs (no ft) to isolate the phase-3 ceiling.

## 5. Interface / BC condition design

- Interface data = frozen-root values + derivatives up to order N−1; well-posedness argument; value-only run as the underdetermination bracket (all losses 1e-12, rel-L2 4.5e-2 — wrong solution).
- `max_interface_derivative_order` ablations 0/1/2/full; bc_max_derivative_order crossed with it; order-1 result: best-ever W0 without wrong-solution modes.
- **Per-order interface-term normalization** (divide order-k term by 1+mean|∂ᵏu0|²) — the breakthrough lever (~7× composed improvement, volume-correlation of laggards eliminated).
- BC-term normalization introduced then **retired** (periodic pairing is scale-free — only interface normalization is meaningful).
- Interface order-decay weighting (decay^k on top of normalization); interface weight ablations (1 vs 1e3; 1e-4 arms); weighted interface data; interface_ic/interface_bc split.
- Flux-form/cPINN-style matching (KdV conservation form, μ²-scaled u_xx) — designed, kept as ablation/aside.
- Annealing of derivative-term weights after basin lock-in — designed, held.
- Rejected with evidence: LRA at interface granularity; asymmetric well-posed BC counts (Bona-Sun-Zhang).
- IC weight ladder 1 → 50 → 100 → 1000 with literature survey (Wight&Zhao, CausalPINNs, PirateNet/SOAP, Kiyani).

## 6. Capacity and expert-architecture studies

- Width/depth grid across eras: 3x20, 3x24, 3x30, 3x32, 3x35, 3x40, 3x50, 3x60…3x80, 4x20, 4x30, 4x35, 4x64, 24-wide vs 30-wide vs 40-wide.
- Capacity-scaled expert sizing via architecture bank (metric/threshold ratio → architecture); fixed-size vs scaled; 150-expert × 2k-weight stress test.
- Expert body types: MLP / ResNet / PirateNet (+ RWF).
- Surgical capacity pairs (same tree, wider experts; equal-M20 more-leaves) — capacity ruled out for W0 in the f64 era; revisited in f32 (17×3x30 vs 11×3x40 count-vs-width trade).
- The capacity-vs-interface-terms attribution measurements (W0 apportionment); laggard = big-box volume correlation +0.92; gap = budget not ceiling (42k→80k: 12.5×→3.8×).
- Adaptive capacity explicitly declared out of scope.

## 7. Collars, overlap, and PoU blending

- Fixed collar: sigma_fraction sweeps 0.03/0.04/0.05/0.06/0.07/0.1/0.2/0.3 (0.2 = soft-blend collapse, measured).
- **Adaptive collar sizing**: per-face width from root-gradient quantiles (q 0.5–0.9), min/max clamps; gradient-threshold rule ("collar ends where |∇u|≈0"); adaptive-collar fine-tune cells in the ft24 grid. (AToE-repo-era invention — NCC-PINN was fixed σ=0.2 throughout.)
- Smoothstep order studies (C² vs C⁴; order ≥ PDE order requirement).
- Collar artifact diagnosis: Σψ=1 ⟹ Σψ″=0 in overlaps — composed-residual pathology.
- Soft-vs-hard evaluation gap (phase 3 trains/evals with hard indicators; σ only bites at ft).

## 8. Fine-tuning schemes

- Global ft of the blend under the full PINN loss (SSBroyden f64 / LBFGS f32; no Adam warmup; smaller lr).
- Collar-concentrated collocation (`collar_data_ratio` 0…1.0 sweeps; collar annealing; collar-volume-ratio transfer analysis across PDEs).
- RAD in ft (winning f64 recipe: static phase 3 → σ 0.1 + RAD 0.3 + resample 500 → 5.55e-8 ≈ measurement floor).
- σ × collar × lr factorial grids (f32 and f64); strong-Wolfe vs fixed-step LBFGS cells; Adam ft (tried, reverted); SSBroyden-f64 vs Adam-f32 ft comparison.
- ft from pretrained experts vs straight from root (skipping phase 3); phase-3-only runs.
- Joint full-domain ft = destructive (rel-L2 ×2000, loss-is-a-broken-proxy evidence, OOM) → single_corrector ft as the fix.
- L2-SP anchoring (tried, unenthusiastic); continuity constraint (rejected); reusing the phase-3 Hessian (dropped); LoRA ft.
- ULMFiT/EWC/gradual-unfreeze/glue-only options surveyed in fine_tune_options_plan.md ("fine-tuning makes it worse" failure mode documented there).
- Patience policy evolution: 5 → 10 → global 10; interval patience; plateau-triggered stop; no-patience runs; FreezeStop guard.

## 9. Optimizers and optimization forensics

- Recipes: 1–2k Adam + 40–80k SSBroyden (f64); 30k scheduled Adam + 30k LBFGS lr 0.1 (f32 "paper" chain); 50k flat Adam + LBFGS (stronger f32 chain); Adam-only 200k; SOAP (jaxpi config); Kiyani SSBroyden reproduction.
- **SSBroyden tolerance-freeze diagnosis** (tolerance_grad 1e-10 → 1e-14) and the **strong-Wolfe line-search rejection wall** (bracket-width × step < 1e-9 hard-coded; scale-invariant; you refused patching scimba).
- **LBFGS instant-freeze forensics**: loss-cliff at the Adam handoff, cubic-zoom collapse returning t=0, draw-dependent; 1-D line-slice probe methodology; fix lr 0.1; fp32/CUDA/determinism exonerated.
- Multi-optimizer (per-expert-group SSBroyden, block-separability verified by zero cross-gradients) — implemented, negative, retired. Hk-refresh (refresh_2nd_order_optimizer) — implemented, negative, retired.
- Adam-warmup length bracketing 0/500/1k/2k/5k/10k (lr-coupled; "never cut epochs and lr together").
- Scheduler ablations (flat vs warmup+exp decay); long-Adam-basin hypothesis (LBFGS can't improve a 200k-Adam root); loss-×1e6 freeze workaround idea (dropped).
- SSBroyden memory law: update peak ≈ 6×8×(Σparams)² → ~20-leaf A10G ceiling; impossible >~20k params; LBFGS as the only 2nd-order option for wide roots.
- rel-L2 ≈ 0.1·√(train loss) empirical law across roots and experts.
- SSBroyden phase not env-reproducible (A100 vs local diverge; Adam bit-identical).

## 10. Sampling and resampling

- RAD (residual^2-based) sampling: roots, phase 3 (localized per-expert design), fine-tune; ratios 0.3/0.5; phi exponent choice.
- Resampling-memorization "sawtooth" diagnosis (on/off-draw 50–250× loss gap); static-draw vs resample-every-{1, 500, 1k, 2k}; verdict: redraws kill shallow phase 3, win in ft from converged starts.
- Per-optimizer sampling semantics (Adam/SOAP fresh minibatches vs quasi-Newton fixed full batch); fixed-Adam-sampling roots; batch decoupling (pool vs minibatch, 4096 floor).
- Collar-concentrated draws as solution-driven counterpart to RAD; causal training considered and rejected for split experts.

## 11. Roots and network features

- Root scaling story: 3x20 → 4x60/5x60/5x64 (f64) → 5x128/6x128/6x256 + RFF (f32); weaker-root "C" experiment (does a weaker root make the experts' job easier).
- Random Fourier features (scale 2, fixed vs auto dim); exact periodic Fourier embedding (root-only vs experts question; BC terms auto-disabled under periodic); RWF layers; PirateNet option; LS output-layer init + hardening.
- Time-marching roots (KdV 3 / KS 5 windows, IC handoff); root-never-retires window variant; window support leakage bug; KS domain cut T=1.0→0.5.
- Root IC-weight and tolerance studies; roots beating published baselines (KdV W0 root 2.8e-8/9.97e-9 vs PirateNet/SOAP 3.4e-4) as the justification for keeping the recipe simple.

## 12. Precision and ground-truth fidelity

- Dual method lines: f64+SSBroyden (2nd-order trio) vs f32+LBFGS/Adam (higher-order KdV/KS).
- Dataset precision fix (generate_dataset followed f32 casts; ~6e-8 rounding in IC targets — same order as the W0 floor); stale-cache loud crash; true-f64 datasets.
- f32/f64 window-edge IC-grid bug; GT self-convergence 4.85e-8 as the measurement floor bounding all sub-5e-8 claims; paper-grade spectral references (ETDRK4, split-step, Cole-Hopf) + solver self-convergence verification.

## 13. Failure anatomies (thesis "analysis" chapters)

- W1 expert-5 propagation failure into a wrong low-residual solution (Daw/Krishnapriyan class); offset hypothesis refuted 1000×; emergent-complexity profile explanation; rescued by fine-tune.
- W0 middle-expert gap apportionment (interface terms exonerated at 1e-12; capacity/optimizer attribution chain culminating in the tolerance-freeze discovery).
- Laggard anatomy: big-box volume correlation, budget-not-ceiling, init/Adam damage, σ=0.2 collapse, ic-100 and iface-order-2 exonerated, soliton-corridor trainability vs root-error-mass separation.
- Fine-tune loss explosions and "objective vs optimizer problem" framing; loss-as-broken-proxy evidence (all terms small, rel-L2 2.8e-2).

## 14. Benchmarks, baselines, reproductions

- Active PDE suite: Burgers1D, Schrödinger/NLS, Allen-Cahn (2nd-order, f64) + KdV (order 3), KS (order 4) (higher-order, f32).
- Removed problems (NCC-PINN implemented, AToE dropped): Wave1D, Burgers2D, Fisher-KPP, convection-diffusion.
- Reproductions run as baselines: PirateNet KdV, Kiyani et al. SSBroyden pipelines, jaxpi SOAP configs, Raissi references.
- Paper-run reproduction methodology (old-commit worktree vs HEAD arms, tree-grid drift, config archaeology); the two "marked" hash-verified citable chains (f32 50k-Adam chain 1.3525e-2; 30k+30k paper chain 5.376e-2).

## 15. Evaluation methodology and infrastructure

- rel-L2 unified onto the solver native grid (full-repo audit; removal of interpolation-fake errors); per-segment best checkpoints; per-expert diagnostics (per-expert rel-L2, split loss terms, capacity/error maps, soft-weight plots).
- Paper figure pipeline (decomposition demos, collar heatmaps, capacity maps, error heatmaps, concatenated training curves, epochs-vs-optimizer-steps axes).
- Run infrastructure: AWS A10G/EC2, Colab A100/L4 runners, screen sessions, OOM/VRAM budgeting tables, dry-run plan validators.

---

## 16. Honest accounting — what I could and could not find

**Found and well-evidenced:** everything above traces to concrete commits, branches, docs, plan cells, transcripts, or memory notes across the four layers.

**Could NOT be recovered:**

1. **NCC-PINN-era chats do not exist on this machine.** The Claude Code project folders for NCC-PINN and Master contain zero transcripts. Everything from Nov 2025 → Jul 2026 (ANT birth, splitting-criteria searches, splits-on-the-go, why decisions were made) is reconstructed from the repo alone. If those conversations happened in Cursor or claude.ai web, I cannot read them.
2. **AToE chats before Jul 27, 2026 are also missing** — the repo starts Jul 4, transcripts start Jul 27. The owner-imitator (Jul 9–13) and Schwarz (Jul 12) eras therefore have *code and tex evidence but no conversational record*; in the surviving chats they appear only as retired references.
3. **Why ANT was abandoned** — single commit "removing the branch" (Feb 19), no written rationale anywhere.
4. **Quantitative outcomes for most NCC-PINN experiments** — commit messages record configs, not results; the results tex is an empty skeleton. Which criterion/variant "won" each NCC-PINN era is not recorded in-repo.
5. **Owner-imitator and Schwarz quantitative comparisons** — runs existed per commit messages, but their output folders were not inventoried; no recorded numbers found.
6. **`lora_tests` outcome** — one commit, no result record yet.
7. **The thesis background/theory section** (Tree-Besov, m-term approximation) referenced by 3_method_updated.tex is not in either repo.
8. **Transcript-mining caveats:** no session-summary records existed (all mining is from user messages); assistant-only topics could in principle be missed (keyword cross-check found none); user messages >1500 chars were truncated during extraction (pasted logs/plans).
9. **Exact content of "7 SOTA PINN features"** (NCC-PINN commit 48520e3) — inferred (causal/LRA/RWF/FF/adaptive-sampling/grad-clip/warmup), not verified against the diff.
10. **Duplicated commit pairs** in NCC-PINN history (rebase/cherry-pick artifacts) were not disentangled; the `AToE-New` branch's distinct purpose vs its siblings is unclear.
