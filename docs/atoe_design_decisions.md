# AToE — Algorithm Design Decisions (from code-review Part D discussion)

Date: 2026-07-09. Scope: design-level changes on top of the clean-run pipeline (bugs/efficiency from Parts A–C handled separately).

> **Status (2026-07-09, branch `AToE`):** D1, D2 (incl. the soft-PoU reporting change), and D3 implemented, always on (`per_leaf_normalization: true` flag exists for D1 ablation). D7 (`adaptive_pinn.split_icbc.interface_decrease_weight`) and D6 (`adaptive_pinn.fine_tune.collar_data_ratio`) implemented, configurable, default 0 (off). All mechanisms log their trigger/data/result (`[Norm]`, `[SplitData] face=... train_box=...`, `[InterfaceAnneal]`, `[Resample-Collar]`) and were verified by unit tests + allen_cahn and KdV time-marching smoke runs.

---

## Decided — to implement

### D1. Per-leaf input normalization
Map each expert's inflated box to $[-1,1]^{d+1}$ before the MLP (affine rescale in `forward_single_expert` / composition).

**Why:** deep leaves receive tiny coordinate ranges (~10⁻²·domain), which conditions a tanh MLP poorly. FBPINN (Moseley et al., 2107.07871) applies separate per-subdomain input normalization precisely for this reason. ~5-line change, largest payoff-per-effort on the list.

### D2. Train experts on region ∪ collar (no POU during expert training)
Enlarge each expert's *training* domain from the hard box $\Omega_j$ to the inflated box $\tilde\Omega_j = \Omega_j \pm \delta$ (the exact support of its window $\Psi_j$), and move the interface faces to the outer edge $\partial\tilde\Omega_j \cap \mathrm{int}(\Omega)$, still with value targets minted from the frozen root $u_0$.

**Key point:** experts remain fully independent — each solves its own closed local problem on a slightly bigger box; the POU never enters the training loss. Agreement on overlaps is a consequence, not a constraint: in a collar $C = \tilde\Omega_i \cap \tilde\Omega_j$ both experts independently approximate $u$, and since weights sum to 1,
$|u_\theta - u| \le \max(|u_i - u|, |u_j - u|)$ on $C$ (overlapping-Schwarz argument). This removes the current mismatch where experts are evaluated (via the POU) exactly where they were never trained, and shrinks the $\partial\Psi\cdot(u_i - u_j)$ cross terms that currently dominate fine-tune.

**Reporting change:** during the expert-training phase, evaluate and log rel-L2 under the *blended POU composition* (not hard indicators), so the metric curve matches the final model's composition throughout (resolves the C9 mixed-regime issue).

### D3. Richer interface conditions (mimic the global PDE structure)
Interior faces close with value **plus derivative** matching against the frozen root, up to the orders the PDE demands (XPINN-style; ceur-ws Vol-2964, arXiv 2104.10013). Where the global BCs are defined through derivatives (e.g. periodic pairing of $u$ and $u_x$), the minted interface data mimics the same structure on root predictions.

**Why:** value-only Dirichlet under-constrains 3rd/4th-order local problems (KdV formally needs 3 conditions per x-face). Expected payoff concentrated on KdV/KS; minor for Burgers/Allen–Cahn.

### Fine-tune stays, but demoted from "repair" to "polish"
With D2, the global fine-tune is no longer structurally necessary to fix collars. It is retained because it removes the two residual error sources no local pass can: (a) the root-error floor inherited through the $u_0$-minted Γ-targets, and (b) nonlinear blending cross terms ($u_\theta\partial_x u_\theta \neq \sum w_j u_j \partial_x u_j$), which D2 makes second-order small but nonzero. Ablation plan: report rel-L2 at (i) blend before fine-tune, (ii) short fine-tune, (iii) full schedule — expect (i) already strong and (ii) ≈ (iii), justifying a reduced fine-tune budget.

---

## Options moving forward — implement both, configurable, default off

### D7. Interface-weight anneal — param `interface_decrease_weight`
Decay $\lambda_\Gamma$ over expert training (schedule controlled by `interface_decrease_weight`). Rationale: Γ-targets carry root error; with fixed $\lambda_\Gamma = 1$ experts stay pinned to that error level even when their residual loss is far below the interface MSE. Consistent with penalty-annealing in DD-PINN practice (APINN). Partially overlaps with D3/D4 (all attack the root floor), hence optional.

### D6. Collar-focused fine-tune sampling — param `collar_data_ratio`
Fraction of fine-tune collocation points drawn from the collar overlaps (where ≥2 windows are active; mask available from `BatchedIndicators`): `1` = all points in collars, `0` = plain uniform sampling. Rationale: flat-top interiors are already solved by the expert phase; the fine-tune's job is reconciling collars, so concentrate the budget there. Sampler-only change.

---

## Considered and dropped

- **D5 / B3 — per-expert optimizers.** At the target scale (5–10 experts, 3×20–30 MLPs ≈ 15k joint params) dense joint SSBroyden is comfortably feasible; the vectorization effort isn't justified. Revisit only if M grows.
- **POU during the expert phase (joint soft training).** Destroys loss separability and the parallel-training claim, forces one $(\sum P_i)^2$ quasi-Newton matrix, and its benefit — training under the evaluation composition — is ~90% covered by D2 + D6 + the existing fine-tune (which *is* joint POU training). FBPINN trains jointly but with Adam for exactly this scalability reason. At most a future ablation ("AToE-joint").
- **D8 — residual-aware M allocation across windows.** Deferred; out of scope for the current experiments.
- **D4 — Schwarz interface refresh.** Not in the current batch; noted as a future option for cases where the root is the bottleneck (KdV μ=0.022, KS), largely subsumed if D3 + fine-tune suffice.

## References
FBPINN — Moseley, Markham, Nissen-Meyer (arXiv 2107.07871; ACOM 2023) · XPINN interface conditions — CEUR Vol-2964; parallel XPINN arXiv 2104.10013 · APINN penalty annealing — EAAI 2023 · DD-ML survey — arXiv 2312.14050 · SSBroyden scaling — Kiyani et al. (arXiv 2501.16371), Urbán et al.
