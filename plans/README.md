# Paper run plans

Runnable plans mirroring the NCC-PINN runs selected for paper reporting,
translated 1:1 (architectures, optimizers, LR schedules, epochs, sampling)
into this repo's config schema. Run with:

```bash
python run_experiments.py plans/<plan>.yaml
```

| Plan | Source run (NCC-PINN) | What it is |
|------|----------------------|------------|
| `allen_cahn_root_creation.yaml` | `allen_cahn_bigger_root_creation_20260630_055714` (70-wide variant) | Vanilla root `[2,70,70,70,1]`, 35k epochs, Adam→SSBroyden@5000, lr 5e-3 exp 0.98/1000 |
| `allen_cahn_M5_paper.yaml` | `allen_cahn_tests_20260703_143902-CURRENT_PAPER` | AToE-Leaves M=5 from the 70-wide root, experts `[2,30,30,30,1]`, 35k+35k epochs |
| `burgers_root_creation.yaml` | `burgers_root_creation_20260704_065044` (60-wide variant) | Vanilla root `[2,60,60,60,60,1]`, 40k epochs, Adam→SSBroyden@1000, RAD sampling, 10k residual pts |
| `burgers_M10_paper.yaml` | `burgers_M10_20260704_151623` | AToE-Leaves M=10 from the 60-wide root, experts `[2,20,20,20,1]`, 40k+40k epochs (source native-grid rel-L2: 1.43e-7) |
| `kdv_root_creation.yaml` | `experiments_plan.yaml @ kdv_root_creation` | JAX-PI/PirateNet recipe roots (MLP/ResNet/PirateNet, 9×256), 2e5 Adam steps, causal + RWF + periodic FF + LS-init |

## Prerequisites / notes

- **Pretrained roots**: `burgers_M10_paper` needs `roots_checkpoints/burgers_root.pt`
  (in the repo). `allen_cahn_M5_paper` needs `roots_checkpoints/allen_cahn_root.pt`
  — produce it from the `allen_cahn_root_creation` run, or convert the NCC-PINN
  checkpoint with `scripts/convert_root_checkpoint.py`.
- **Patience translation**: the sources used `patience_epochs: 5000` on the train
  loss; this repo uses `patience_evals` on the eval physics loss. All plans set
  `eval_every: 1000` × `patience_evals: 5` = the same 5000-epoch window.
- **Sampling translation**: sources that used `initial_train_ratio: 0.026` are
  translated to explicit counts (`0.026 × n_residual`, e.g. 106 for allen_cahn,
  260 for the burgers root).
- **Dataset cache**: `datasets/<problem>/` is cached by existence — after changing
  any `sampling.*` value, delete that problem's dataset folder so it regenerates.
