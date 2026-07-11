# Phase 3 / Fine-Tune: Why `main` Wins Locally and Loses Globally — Diagnosis and Literature-Based Directions

*Working note, 2026-07-11. Based on: `main`-branch run `allen_cahn_M8_20260709_090246` (training log analyzed in full),
AToE-branch rounds 1–2 (A/B/C, R1–R4), and the method write-up diff (`docs/3_method_updated.tex`).*

---

## 1. What the main-branch log actually shows

| Stage | Metric (rel-L2, grid) | Note |
|---|---|---|
| Root (45k = 5k Adam + 40k SSBroyden) | **3.00e-5** | best restored @34k |
| End of phase 3 (45k, **hard indicators**) | **3.42e-6** | ~9× better than root |
| Fine-tune start (epoch 90001, **soft PoU**) | **8.4e-2** | the same weights, only the evaluation composition changed |
| Fine-tune Adam stage (5k @ lr 1e-4) | 8.4e-2 → **2.4e-1** | Adam *degrades* the good init |
| Fine-tune best (SSBroyden, @113k) | **3.32e-5** | ≈ root; the phase-3 gain is fully erased |

Three separate failure mechanisms are visible, and they are worth keeping distinct:

1. **The blend shock.** Under `main`'s phase 3 each expert solves a well-posed local BVP on its *hard*
   region (true IC/BC where applicable, u0 interface data, continuity pairs) — and it genuinely solves it
   (3.4e-6 composed with hard indicators). But the smoothstep composition evaluates each expert throughout
   its collar, i.e. *outside* the set it was ever trained on. Two experts that each carry ~1e-6 error inside
   their own regions can disagree at O(1e-1) in a collar neither of them saw. The 3.4e-6 → 8.4e-2 jump at
   the phase3→fine-tune boundary is exactly this extrapolation error, not an optimization artifact.
   (Same phenomenon at spawn: the continuity check reports a 64% output-norm change.)
2. **Fine-tune as structural repair.** From an 8.4e-2 start, the global fine-tune is not polishing — it is
   re-solving the problem in the collars. It plateaus at root level (3.3e-5). The information that made the
   experts 10× better than the root is *local* (sub-collar-scale detail), and the joint global loss with 10k
   uniform points has no mechanism to preserve it while repairing O(1e-1) collar mismatch.
3. **A reporting trap.** Phase-3 best-checkpoint selection on `main` used hard indicators, i.e. it optimized
   and selected a *different object* than the one that is reported at the end. The 3.4e-6 is real, but it is
   the accuracy of the hard patchwork, not of the deliverable u_θ.

## 2. What the AToE branch fixed — and what it traded away

The D-changes remove failure #1 completely (experts trained on the full window support Ω̃_j, soft-eval
checkpointing, per-leaf normalization): there is no blend shock, and fine-tune is a genuine polish. But the
closure redesign replaced two *exact* couplings with one *approximate* one:

- `main`: true IC/BC per boundary expert + periodic pairing + neighbor continuity pairs + u0 only on interior faces;
- `AToE`: frozen u0 guides on **all** outer faces (including t=0 and ∂Ω), exact IC/BC only through the
  composition terms.

Every expert is therefore pinned, on the entire boundary of its training box, to data carrying the root's
3e-5 error. The u0 guide is the *only* face information an interior expert gets, and it is frozen. That is
an accuracy ceiling at root level by construction; annealing (D7) lifts the ceiling but removes the
coupling that made the local problems well-posed, so it trades bias for variance rather than removing the
limit. Result: ~1.67× over root instead of ~9×.

**The research question, sharply:** get `main`'s local accuracy (exact, mutually-consistent local problems)
while training/evaluating the object we report (the smooth PoU composition), so the fine-tune never has to
repair anything.

---

## 3. Mechanisms in the literature

### A. Train the *composition's* residual in the collars (FBPINN-style, but only where it matters)
FBPINN ([Moseley, Markham, Nissen-Meyer 2023](https://link.springer.com/article/10.1007/s10444-023-10065-9);
multilevel version [Dolean, Heinlein, Mishra, Moseley, CMAME 2024](https://www.sciencedirect.com/science/article/pii/S0045782524003724))
never trains experts on local problems at all — the PDE residual is always applied to the PoU-composed
output, so there is no notion of blend mismatch. The cost is losing the "easy localized problem" property.
**Key structural fact for AToE:** with flat-top windows, u_θ ≡ u_j outside the collars. So a composed-residual
loss differs from the per-expert residual loss *only at collar points*, where ≤2–3 experts are active. A hybrid
is therefore natural and cheap:

> keep `main`'s per-expert loss (hard region, true IC/BC, interface data) **and add the residual of the
> blended u_θ on collar collocation points** during phase 3.

This is exactly "involving the smoothstep in the main-branch loss": the cross terms that fine-tune currently
repairs become part of the expert-training objective, gradients still flow only to the few experts active in
each collar, and the phase-3 metric = the reported metric. D6 (collar sampling) becomes a phase-3 feature
rather than a fine-tune patch.

### B. Refresh the interface data (Schwarz alternating) instead of freezing u0
The frozen-root ceiling is the textbook motivation for Schwarz iteration: solve locals with interface data
from the *current* neighbor solution, re-exchange, repeat — convergence does not depend on the accuracy of the
initial guess. Applied to PINNs: [Snyder et al., "Coupling of PINNs via the Schwarz alternating method"
(arXiv 2311.00224)](https://arxiv.org/pdf/2311.00224); DeepDDM / D3M lines; a
[quasi-optimal DDM for NN-based Schrödinger solvers (CPC 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0010465524000523);
and non-overlapping Schwarz-type variants with generalized (Robin-type) interface conditions
([PECANN-DD, arXiv 2409.13644 / CMAME 2025](https://arxiv.org/abs/2409.13644)).
**AToE translation:** keep the current face-guide machinery but re-mint the face targets every resample
interval (500 epochs) from the current neighbor expert / current u_θ instead of the frozen root, optionally
starting from u0 (iteration 0 ≡ current scheme). This removes the 3e-5 ceiling without giving up
well-posedness — it is the principled version of what D7's anneal gestures at.

### C. Symmetric neighbor-continuity penalties (XPINN/cPINN transmission conditions)
[cPINN (Jagtap, Kharazmi, Karniadakis 2020)](https://doi.org/10.1016/j.cma.2020.113028) and
[XPINN (Jagtap & Karniadakis 2020)](https://doi.org/10.4208/cicp.OA-2020-0164) close local problems without
any oracle: value continuity + flux/derivative continuity (+ optionally residual continuity) between the two
experts sharing an interface, penalized symmetrically so both sides move. `main` already had continuity
pairs; `AToE` dropped them in favor of u0 guides. Bringing them back *in the collars* (where both windows are
active, targets = each other, not u0) is the static-penalty cousin of (B): no frozen error source, experts
co-adapt. Robin-type combinations (value + α·derivative) are reported more robust than pure Dirichlet
exchange in the DD-PINN literature above.

### D. Defect correction: keep the root in the composition, let experts learn the *error*
Reformulate u_θ = u0 + Σ_j Ψ̃_j δ_j with the root frozen (or slowly unfrozen in fine-tune). This is the
multi-stage / boosting mechanism — successive networks fit the previous stage's residual error and can push
many orders of magnitude below the first stage
([Wang & Lai, multi-stage neural networks, arXiv 2311.xxxx / JCP 2024](https://arxiv.org/html/2407.17213v1)
and the spectrum-informed follow-up), and it is also how multilevel FBPINNs treat coarse levels — they are
*kept* in the composition, not retired
([Dolean et al. 2024](https://www.sciencedirect.com/science/article/pii/S0045782524003724)).
Structural advantages for AToE specifically:
- at spawn, δ_j = 0 reproduces the root exactly → no spawn discontinuity (the 64% warning disappears), no
  blend shock ever, every stage starts from 3e-5 rather than from O(1);
- the tree was *built* from the root's wavelet detail — the experts' regions are literally localizations of
  the root defect, which is now exactly their regression target;
- interface guides become trivial: the correct face value for δ_j near a face where the root is trusted is ~0.
Cost: the root stays in inference, weakening the "lighter network" story (though it is one shared frozen
network, and the small-experts argument survives as "cheap correction capacity").

### E. Remove the IC/BC competition entirely: hard constraints
FBPINN enforces outer BCs by construction (constraining operator); distance-function-based hard imposition is
standard ([Sukumar & Srivastava, CMAME 2022](https://doi.org/10.1016/j.cma.2021.114333)); and for the periodic
set (Allen–Cahn, KdV, KS, Schrödinger) the SOTA configs in `docs/pde_benchmarks.md` (Kiyani/SSBroyden, vRBA)
use **periodic feature embeddings** — periodicity in u and u_x holds exactly, the bc loss term vanishes.
With hard IC/BC there is nothing for the composition-level IC/BC terms to fight the face guides over, and the
x-face derivative guides (D3) become redundant for periodicity. Orthogonal to A–D and cheap.

---

## 4. Suggested priority (no code changes yet)

1. **A — collar-composed residual on top of the `main`-style local loss.** Directly attacks the observed
   failure (blend shock / fine-tune repair) while keeping everything that produced 3.4e-6. Most likely to
   preserve the ×10. Sanity metric: soft-eval rel-L2 at end of phase 3 should match the hard-eval number.
2. **B (or C) — replace frozen u0 face data with refreshed neighbor/composition data (or symmetric
   continuity).** Attacks the root-error ceiling, which becomes the binding constraint once (1) works.
   B is a small change to the existing target-minting code path (re-mint every resample).
3. **D — defect-correction composition.** Bigger reframe; test as an ablation arm. If (1)+(2) stall above
   ~1e-5, this is the mechanism with literature precedent for multi-order-of-magnitude stacking gains.
4. **E — hard/periodic IC-BC.** Independent simplification; consider bundling with any of the above.

Also worth fixing regardless of direction: phase-3 checkpoint selection must score the soft composition
(already the case on AToE branch — keep it), and the fine-tune Adam warm-up should stay removed (the log
shows Adam @1e-4 actively destroying a good init before SSBroyden recovers).

---

### Sources
- Moseley, Markham, Nissen-Meyer — FBPINNs, *Adv. Comput. Math.* 2023
- [Dolean, Heinlein, Mishra, Moseley — Multilevel DD-based architectures for PINNs, CMAME 2024](https://www.sciencedirect.com/science/article/pii/S0045782524003724)
- [Snyder et al. — Coupling of PINNs via the Schwarz alternating method, arXiv:2311.00224](https://arxiv.org/pdf/2311.00224)
- [PECANN-DD — Non-overlapping Schwarz-type DDM with generalized interface conditions, arXiv:2409.13644 / CMAME 2025](https://arxiv.org/abs/2409.13644)
- [Quasi-optimal DDM for NN computation of time-dependent Schrödinger, Comput. Phys. Commun. 2024](https://www.sciencedirect.com/science/article/abs/pii/S0010465524000523)
- Jagtap & Karniadakis — XPINN, *Commun. Comput. Phys.* 2020; Jagtap, Kharazmi, Karniadakis — cPINN, CMAME 2020
- Hu et al. — APINN (adaptive/annealed gated DD), 2022 (already cited in the method write-up)
- [Wang & Lai — Multi-stage neural networks: function approximators of machine precision (and spectrum-informed follow-up, arXiv:2407.17213)](https://arxiv.org/html/2407.17213v1)
- [Sukumar & Srivastava — Exact imposition of boundary conditions via distance functions, CMAME 2022](https://doi.org/10.1016/j.cma.2021.114333)
- `docs/pde_benchmarks.md` — SSBroyden (Kiyani et al. 2025) and vRBA (Hag et al. 2025) configs with periodic embeddings
