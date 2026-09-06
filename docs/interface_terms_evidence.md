# Interface loss terms — evidence map for the thesis

Compiled 2026-09-06 from the saved run folders (`outputs/experiments/` and the Desktop archive), the saved `config_used.yaml` of every KdV/KS run (332 runs tabulated), git history, and the session decision notes. Covers the chapter on interface derivative order (`max_interface_derivative_order`), per-order normalization (`interface_term_normalization`), the retired knobs (`interface_order_decay`, `bc_term_normalization`), and the IC/BC weight ladder.

Two premises worth stating up front:

- There was **never a KS phase-3 (experts) run**. Every KS run in the repo is roots only. The KS cell of the 2026-07-21 batch (`C_ks_M30_tm5`) was launched on 2026-07-22 but no output survived.
- The **final f32 pipeline caps the periodic pairing at order 1 too**, not just the interfaces (see §5).

---

## 1. Value-only interface runs

**Which run.** The value-only interface run is the 3-window KdV run

```
Desktop\old_bad_kdv\experts_creation_sch_kdv_ks_20260721_045357\kdv-base-2-60-60-60-60-1-experts-2-20-20-20-1-tanh\20260721_092724
```

at commit `32c1cac`. Git confirms it predates derivative matching: commit `5c2352a` (2026-07-24, "derivative bc in all pdes that where missing") is the one that added to `losses/split_loss.py` the matching of "spatial derivatives up to order m-1 at interface points … (value-only under-determines KdV/KS)". Config: M=20 split [2, 6, 12] over 3 windows, ic = bc = 1, experts 3x20, 45k epochs per segment (Adam → SSBroyden at 5k), float64.

**Window 0 (the clean bracket).** Windows 1 and 2 inherit a wrong IC from window 0, so their roots are already at 0.29 / 0.34 rel-L2 and are unusable as evidence.

| Window 0 | rel-L2 |
|---|---|
| root | 8.61e-8 |
| expert 0 (right edge: true periodic BC + one interface) | 3.7e-7 |
| expert 1 (left edge) | 6.2e-7 |
| expert 2 (middle, x ∈ [-0.46, 0.53], holds the soliton) | **4.53e-2** |
| composed phase 3 | 3.22e-2 |

The middle expert's final loss terms were all at machine noise: residual 8e-12, IC 2e-12, interface 4e-13, total 1.1e-11. That is the underdetermination signature — a *different* KdV solution that satisfies the value data on both faces. KdV needs 3 boundary conditions; value-only on two faces supplies 2.

**What the prediction looks like** (`window_0/adaptive_plots/pred_after_phase3_best_ep67000_relL2_3.22e-02.png`): the soliton steepening near x ≈ 0.5 at t > 0.2 is missing from the prediction; the absolute-error map shows dispersive fringes fanning from the x = 0.53 face into the interior, starting near t ≈ 0.05 and growing toward the top of the window (error 1e-1 class at t > 0.2). The face itself is pinned exactly (zero error on the interface line).

**Direct same-config value-vs-derivative comparison** (f32, 2026-08-15, `outputs/experiments/kdv_adam200k_phase3_bcu_iface0_vs_iface1_20260815_153555`; 200k Adam only, bc order 0, 17 experts 3x30, interface normalization on):

| interface order | best phase-3 rel-L2 |
|---|---|
| 0 (value only) | 2.69e-1 |
| 1 (value + u_x) | 3.97e-2 |

---

## 2. Per-order normalization ablation

KdV only.

**The 2026-08-14 runs** (`outputs/experiments/kdv_ifnorm_M18wtree_vs_M12plain_20260814_102953`) are f64, full domain, full interface order, ic = 100. Their "before" is the plain-M20 full-domain no-norm run of 2026-08-13 (`kdv_fulldomain_M20_plain_vs_wtree_20260813_180931`). The pair is **not the same M**, so quote it as an approximate factor:

| run (f64, full domain, full order) | normalization | rel-L2 |
|---|---|---|
| plain M20, 2026-08-13 | off | 3.42e-5 |
| plain M12, 2026-08-14 | on | 6.74e-6 |
| wtree M18 W3-linear, 2026-08-14 | on | 5.19e-6 |

Improvement ≈ 5x–6.6x. Side findings from that analysis (memory note 2026-08-14): the volume–error correlation of laggards vanished (+0.92 → ≈0), the new laggards are the t > 0.67 band, both runs were budget-stopped still descending ~9 %/1k.

**The exact same-config ablation** is f32, 2026-08-17:

```
Desktop\kdv_p3_from30k30kroot_iface_x_bc_ablation_20260817_131512   (norm ON)
Desktop\kdv_p3_from30k30kroot_iface_x_bc_ablation_20260817_153803   (norm OFF)
```

Four cells each, same root (`kdv_adam30-lbfgs30_full_batch_root_periodic_128x6.pt`, the periodic-Fourier-embedding root, rel-L2 5.32e-3), 60k budget, Adam → strong-Wolfe LBFGS at 30k, weights 1e3, experts 3x30:

| cell | norm off | norm on | factor |
|---|---|---|---|
| iface full, bc 2 | 1.04 (loss stuck at 4.3e5, froze at 30100) | 7.25e-2 | ~14x |
| iface full, bc 0 | 1.04 | 5.67e-2 | ~18x |
| iface 1, bc 2 | 3.52e-1 | 2.40e-2 | ~15x |
| iface 1, bc 0 | 3.46e-1 | 1.03e-2 | ~34x |

Without normalization the full-order cells never left the initial dxx-dominated loss (init scale of the dxx interface term is 1e5–3.8e6). Caveat for citation: these p3 runs sit on the periodic-embedding root, which the final pipeline does not use.

**Mechanism** (code comment, `losses/split_loss.py` ~L107): the orders' natural scales differ by 1e4–1e6, so at uniform weights the u_xx term soaks the interface gradient budget while the *value* term — the one that caps the expert's rel-L2 — is starved. Each order-k derivative term is divided by (1 + mean|∂ₓᵏu₀|²) over the expert's own interface points (frozen root ⇒ constants); the +1 floor means a term is only ever shrunk, never amplified; the value term is unscaled. For 2nd-order PDEs the factors are ≈1 (no-op on the AC/Burgers/Schrödinger benchmarks).

---

## 3. f32 runs where interface/pairing order ≥ 2 breaks the optimizer

**Failure mode**: always the same — an instant strong-Wolfe LBFGS freeze at the Adam → LBFGS handoff, logged as `stop=freeze_stop` at epoch 30100 (switch at 30000) with the train loss stuck at O(1). Order-1 cells run their full budget. Mechanism per the LBFGS-freeze diagnosis (2026-08-17): the first trial step t₀ = lr·min(1, 1/|g|₁) lands on a loss cliff and the cubic zoom returns t = 0; params never move.

**Cleanest matrix**: `outputs/experiments/kdv_p3_bothnorm_iface_x_bc_x_ifacew_20260818_130516` (f32, both normalizations on, weights 1e3, same periodic root as above, 3x30 experts, 60k budget):

| interface order | bc pairing order | stop | best rel-L2 |
|---|---|---|---|
| 1 | 1 | budget (60k) | **1.06e-2** |
| 1 | 2 | freeze at 30100 | 6.58e-2 |
| 2 | 1 | freeze at 30100 | 6.11e-2 |
| 2 | 2 | freeze at 30100 | 7.83e-2 |

Any dxx term — on the interface *or* on the periodic pairing — triggers it. The 2026-08-17 batch (§2) shows the same pattern: full-order cells froze at 30100; iface1/bc2 froze later at 47994 (2.40e-2); iface1/bc0 ran to budget (1.03e-2). A 4x64-expert variant of the iface1/bc1 cell (`Desktop\kdv_p3_bothnorm_iface1_bc1_experts4x64_20260820_045028`) gave 1.01e-2 — capacity did not move it.

**f64 contrast** — the cap is *not* a clear accuracy win there (`outputs/experiments/kdv_p3_trees_from5x60roots_ord1_vs_full_20260825_200910`, ic = bc = interface = 1e3, both norms on, 41k epochs):

| tree | order 1 | full order |
|---|---|---|
| A (M12 W2-linear, 11 leaves of 3x24) | 3.15e-1 — failed: expert 7 (wide late-time interior leaf x ∈ [-0.36, 0.07], t ∈ [0.5, 1]) ended at 1.46, early-stopped at 13k | 4.47e-6 |
| B (M12 W3-linear-zero) | 3.93e-6 | 7.51e-6 |
| C (M15 W3-linear, 3x20) | 6.50e-6 | — |

Roots the same day (5x60, f64): bc order 1 → 2.46e-6, full → 3.05e-6. So the order-1 cap should be presented as an **f32 optimizer fix**, not an accuracy claim; in f64 the full-order terms are load-bearing for well-posedness and the choice is tree-dependent.

---

## 4. `interface_order_decay`

**1.0 (off) everywhere.** The plan base block sets it to 1.0 (`experiments_plan.yaml`, `split_icbc`), and across all 332 tabulated runs the key is either 1.0 or absent (absent = code default 1.0). No saved run ever used a decay < 1. It was implemented 2026-08-14 (commit `ef2556e`, "order-weight decrease on interface") with the measured motivation recorded in `losses/split_loss.py` ~L126 (normalized dx/dxx terms still 40–77 % of laggards' loss while matching the root's derivatives 7–20x tighter than the root matches truth) but was never exercised.

---

## 5. Periodic pairing order and `bc_term_normalization` in the final runs

**Final f32 pipeline (KdV and KS)** — both caps at order 1:

| knob | value |
|---|---|
| `split_icbc.max_interface_derivative_order` | 1 |
| `kdv.bc_max_derivative_order` / `ks.bc_max_derivative_order` | 1 (pairing capped too) |
| `split_icbc.interface_term_normalization` | true |
| `split_icbc.bc_term_normalization` | false |
| `split_icbc.interface_order_decay` | 1.0 |
| loss weights ic / bc / interface_ic / interface_bc | 1e3 each |

Holds for the marked f32 chains: p3 M18 W3-linear 3x35 (2026-09-01, `kdv_f32_p3_from_adam50k_lbfgs01_root_e3x35_20260901_140139`), the windowed-root p3 runs (2026-09-03/04), the 3-window f32 root, the KS w5 T=0.5 roots (`ks_timewindow_root_w5_T05_f32f64_pair_20260903_133238`) and the current plan block. In code the two caps are independent: the interface cap never touches the pairing (`split_loss.py` ~L421, "the pairing's order is capped by bc_max_derivative_order, NOT by the interface cap"), and `bc_max_derivative_order` is the per-problem knob shared by the root loss (`losses/kdv_loss.py` L141, `losses/ks_loss.py` L147) and the split pairing.

**Best f64 chain** (`kdv_f64_repro_bcnormoff_root_p3_M12W2lin_20260827_200114-ROOT+EXPERTS_F64/.../20260827_200117`, root 1.3126e-5, p3 1.3142e-5): full order on both (interface `null`, bc order 2), interface normalization on, **bc normalization off**.

**Why bc normalization is off** (decided 2026-08-26): the periodic pairing only demands u_L = u_R per derivative order — a scale-free constraint; dividing both sides by the same scale changes nothing the term enforces. Interface terms are different: they match the expert's derivative *values* to the frozen root's, whose magnitudes grow steeply with order. Runs from 2026-08-18 to 08-25 (the bothnorm batches, the 30k+30k "paper" chain `kdv_AB_root6x128_then_expertsize_2026082*`) have `bc_term_normalization: true`; treat those as pre-retirement.

---

## 6. IC/BC weight ladder

**Nothing was ever run at 1e4.** The ladder in the saved runs is 1 → 50 → 100 → 1e3:

| weights | period | evidence |
|---|---|---|
| ic = bc = 1 | 2026-07-21 → 08-09, f64 | W1 phase 3 stuck at 4.74e-1 (`Desktop\kdv-base-2-60-60-60-60-1-experts-2-20-20-20-1-tanh\20260809_063705\window_1`, expert 5 at 1.24 — propagation into a wrong low-residual solution; fine-tune rescued it to 5.27e-4) |
| ic = 50 (bc 1), W1 only | 2026-08-10 | `Desktop\kdv-base-...\20260810_062828-SUCCESSFULL_IC-50-EXPERTS\window_1`: W1 phase 3 **4.70e-6** at 42k — wrong-solution class gone |
| ic = 100, bc = 1 | 2026-08-10 → 08-22, f64 | roots ic100 tol14 (W0 9.97e-9, W1 9.40e-8, W2 2.99e-7); order-1 experts on them (`kdv_w012_experts_iface1_20260811_052432`) W0 1.15e-7, W1 1.18e-6, W2 2.84e-6. Measured: ic share of the end-state loss 0–4 %, so 1 → 100 was a no-op on W0 |
| ic = bc = interface_ic = interface_bc = 1e3 | f32 from 2026-08-15, f64 from 08-24 | every final chain, KdV and KS |

The 1e3 value came from the 2026-08-10 literature survey (CausalPINNs: L₀ = 1e3·loss_ics for regular KS, 1e4 for chaotic KS; Wight & Zhao: 100 on the IC term), not from a same-config run that beat 100. **There is no same-config 100-vs-1e3 cell.**

The closest *weight* ablation is on the interface weights: the `ifw1` cells of the 2026-08-18 batch (`Desktop\kdv_p3_bothnorm_iface_x_bc_x_ifacew_20260818_130955`; interface weights 1 with ic = bc = 1e3) all landed at 2.88e-1 – 2.94e-1 versus 1.06e-2 with interface weights 1e3 (~27x).

On KS the weight increases were measured useless in a different sense (2026-09-04): the window handoffs already fit the IC to 1e-9 (handoff jumps ×1.00), and the per-window error growth is the f32 floor, not IC/BC enforcement.

---

## Run index (quick reference)

| purpose | path |
|---|---|
| value-only bracket (KdV, f64) | `Desktop\old_bad_kdv\experts_creation_sch_kdv_ks_20260721_045357\...\20260721_092724` |
| order 0 vs 1, same config (f32) | `outputs/experiments/kdv_adam200k_phase3_bcu_iface0_vs_iface1_20260815_153555` |
| ifnorm f64 runs | `outputs/experiments/kdv_ifnorm_M18wtree_vs_M12plain_20260814_102953`, before: `kdv_fulldomain_M20_plain_vs_wtree_20260813_180931` |
| norm on/off × order, same config (f32) | `Desktop\kdv_p3_from30k30kroot_iface_x_bc_ablation_20260817_131512` / `_153803` |
| iface × bc order matrix, both norms (f32) | `outputs/experiments/kdv_p3_bothnorm_iface_x_bc_x_ifacew_20260818_130516` (+ `_130955` on Desktop for the ifw1 cells) |
| order 1 vs full (f64) | `outputs/experiments/kdv_p3_trees_from5x60roots_ord1_vs_full_20260825_200910`, roots `kdv_roots5x60_ic1kbc1k_bothnorm_ord1_vs_full_20260825_*` |
| ic ladder W1 | `Desktop\kdv-base-2-60-60-60-60-1-experts-2-20-20-20-1-tanh\{20260809_063705, 20260810_062828-SUCCESSFULL_IC-50-EXPERTS}` |
| ic 100 experts / roots | `Desktop\kdv_w0w1_experts_only_ic100_20260810_135925`, `outputs/experiments/kdv_roots_ic100_tol14_20260810_192731`, `kdv_w012_experts_iface1_20260811_052432` |
| final f32 chains | `outputs/experiments/kdv_f32_p3_from_adam50k_lbfgs01_root_e3x35_20260901_140139`, `Desktop\kdv_f32_p3_windowedroot_M18W3lin_e3x50_50a40l_h1000_20260904_085025` |
| final f64 chain | `outputs/experiments/kdv_f64_repro_bcnormoff_root_p3_M12W2lin_20260827_200114-ROOT+EXPERTS_F64` |
| KS roots (f32, w5) | `outputs/experiments/ks_timewindow_root_w5_T05_f32f64_pair_20260903_133238` |
