"""Smart initialization for PINN models.

Strategies (all configurable, architecture-agnostic):
  hidden: glorot  — Glorot uniform + zero bias for all hidden linear layers.
                    Also zero-inits fc2 in ResNet blocks for identity-like start.
  hidden: parent_weights — copy weights from the parent model at expert spawn
                    (handled by apply_parent_copy_init).
  output: zero    — Zero-initialize the output linear layer.
  output: ls      — Least-squares fit of output layer to IC data (base model only).

Reference: Wang et al. (2024) "PirateNets: Physics-informed Deep Learning with
Residual Adaptive Networks." JMLR.
"""

import torch
import torch.nn as nn
from typing import Dict

from utils.logging_config import get_logger

logger = get_logger(__name__)


def _get_output_layer(model: nn.Module) -> nn.Linear:
    """Return the final output nn.Linear for any supported architecture."""
    if hasattr(model, 'output_proj'):
        return model.output_proj          # ResNetModel or PirateNet
    names = model.get_layer_names()
    return model.network[names[-1]]       # FCNet


def apply_hidden_init(model: nn.Module, cfg: dict) -> None:
    """Apply Glorot uniform + zero bias to all hidden linear layers.

    When init.hidden == 'glorot':
    - All nn.Linear / RWFLinear except the output layer get xavier_uniform_ with
      the gain appropriate for the configured activation function.
    - Biases are set to zero.
    - For ResNet: fc2 in every ResBlock is additionally zero-initialized so each
      block starts as an identity map (x + F(x) ≈ x), matching PirateNet's alpha=0.
    """
    init_cfg = cfg.get('init', {})
    if init_cfg.get('hidden', 'default') != 'glorot':
        return

    try:
        from models.rwf_layer import RWFLinear
        linear_types = (nn.Linear, RWFLinear)
    except ImportError:
        linear_types = (nn.Linear,)

    activation = cfg['activation']
    gain = nn.init.calculate_gain(activation)
    out_layer = _get_output_layer(model)

    n_hidden = 0
    for module in model.modules():
        if isinstance(module, linear_types) and module is not out_layer:
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            n_hidden += 1

    # ResNet identity-like start: zero fc2 in every ResBlock after Glorot
    try:
        from models.resnet_model import ResBlock
        n_resblocks = 0
        for module in model.modules():
            if isinstance(module, ResBlock):
                nn.init.zeros_(module.fc2.weight)
                n_resblocks += 1
        if n_resblocks:
            logger.info(f"  [Init] ResNet: zeroed fc2 in {n_resblocks} residual blocks (identity start)")
    except ImportError:
        pass

    logger.info(f"  [Init] Glorot uniform (gain={gain:.4f}) applied to {n_hidden} hidden layers")


def apply_output_init(
    model: nn.Module,
    train_data: Dict[str, torch.Tensor],
    cfg: dict,
    device: torch.device,
) -> None:
    """Initialize the output layer of the BASE MODEL (only).

    Modes (init.output):
      'zero'    — zero weight and bias
      'ls'      — least-squares fit to IC data
      'default' — no-op (keep PyTorch default)

    Raises ValueError if there are too few IC points for a well-determined LS system.
    The ValueError is caught by the atexit emergency handler and saved to metrics.
    """
    init_cfg = cfg.get('init', {})
    output_mode = init_cfg.get('output', 'default')
    out_layer = _get_output_layer(model)

    if output_mode == 'zero':
        with torch.no_grad():
            nn.init.zeros_(out_layer.weight)
            if out_layer.bias is not None:
                nn.init.zeros_(out_layer.bias)
        logger.info("  [Init] Output layer: zero-initialized")

    elif output_mode == 'ls':
        use_bias = init_cfg['ls_use_bias']

        mask_ic = train_data['mask']['IC']
        n_ic = int(mask_ic.sum().item())
        hidden_dim = out_layer.weight.shape[1]
        required = hidden_dim + (1 if use_bias else 0)

        if n_ic < required:
            raise ValueError(
                f"[Init] LS-init requires ≥{required} IC points "
                f"(hidden_dim={hidden_dim}, use_bias={use_bias}), got {n_ic}. "
                f"Increase sampling.n_initial_train (or initial_train_ratio) or disable ls_init."
            )

        x_ic = train_data['x'][mask_ic].to(device)
        t_ic = train_data['t'][mask_ic].to(device)
        h_gt_ic = train_data['h_gt'][mask_ic].to(device)
        inputs = torch.cat([x_ic, t_ic], dim=1)

        model.eval()
        with torch.no_grad():
            _, features = model(inputs, return_activation=True)  # (N, hidden_dim)

        if use_bias:
            H = torch.cat(
                [features, torch.ones(n_ic, 1, device=device)], dim=1
            ).float()  # (N, hidden_dim+1)
        else:
            H = features.float()  # (N, hidden_dim)

        solution = torch.linalg.lstsq(H, h_gt_ic.float()).solution
        # solution shape: (hidden_dim[+1], output_dim)

        with torch.no_grad():
            if use_bias:
                out_layer.weight.copy_(solution[:-1].T)   # (output_dim, hidden_dim)
                if out_layer.bias is not None:
                    out_layer.bias.copy_(solution[-1])     # (output_dim,)
            else:
                out_layer.weight.copy_(solution.T)

        model.train()
        output_dim = out_layer.weight.shape[0]
        logger.info(
            f"  [Init] Output layer: LS-init from {n_ic} IC points "
            f"(hidden_dim={hidden_dim}, output_dim={output_dim}, use_bias={use_bias})"
        )

    # output_mode == 'default' → no-op


def apply_expert_init(expert: nn.Module, cfg: dict, zero_output: bool = True) -> None:
    """Initialize a newly spawned expert network.

    Applies:
    - Glorot hidden init (if init.hidden == 'glorot'), else PyTorch default
    - Output layer based on zero_output:
      - zero_output=True: zero output so the expert starts at u=0
      - zero_output=False: same init as hidden layers

    LS-init is NEVER applied to experts (only the base model gets it).

    Args:
        expert: The expert network to initialize
        cfg: Config dict containing 'init' and 'activation' settings
        zero_output: If True, zero the output layer; if False, use the
                     hidden-layer init for the output layer as well.
    """
    apply_hidden_init(expert, cfg)

    out_layer = _get_output_layer(expert)
    init_cfg = cfg.get('init', {})
    hidden_mode = init_cfg.get('hidden', 'default')

    with torch.no_grad():
        if zero_output:
            nn.init.zeros_(out_layer.weight)
            if out_layer.bias is not None:
                nn.init.zeros_(out_layer.bias)
        else:
            if hidden_mode == 'glorot':
                activation = cfg.get('activation', 'tanh')
                gain = nn.init.calculate_gain(activation)
                nn.init.xavier_uniform_(out_layer.weight, gain=gain)
                if out_layer.bias is not None:
                    nn.init.zeros_(out_layer.bias)
            # else: keep PyTorch default (Kaiming uniform)


def apply_parent_copy_init(
    expert: nn.Module,
    parent_model: nn.Module,
    cfg: dict = None,
    copy_output: bool = False,
) -> None:
    """Copy weights from parent_model into a newly spawned expert.

    copy_output=False: output layer is zeroed after copy so the expert
        contributes u_k=0 at spawn time.
    copy_output=True (AToE-Leaves): output layer is also copied; the parent is
        retired on spawn so children must start from the parent's full solution.

    The FULL module state is copied (not just weight/bias) so factorized
    layers like RWFLinear transfer their scale parameter too — otherwise the
    child's effective weights (scale * weight) would differ from the parent's.

    Raises RuntimeError (stopping training with a clear log) if the expert's
    architecture does not match the parent's — a parent_weights init that
    cannot actually copy would otherwise silently train from a random/zero
    start while claiming to be a copy.
    """
    try:
        from models.rwf_layer import RWFLinear
        linear_types = (nn.Linear, RWFLinear)
    except ImportError:
        linear_types = (nn.Linear,)

    def _copy_module(dst: nn.Module, src: nn.Module) -> bool:
        if dst.weight.shape != src.weight.shape:
            return False
        try:
            dst.load_state_dict(src.state_dict())
        except Exception:
            # Param-set mismatch (e.g. bias presence): copy what matches.
            dst.weight.data.copy_(src.weight.data)
            if dst.bias is not None and src.bias is not None:
                dst.bias.data.copy_(src.bias.data)
        return True

    out_layer_new = _get_output_layer(expert)
    out_layer_par = _get_output_layer(parent_model)

    # Collect hidden layers only (exclude output) for both expert and parent.
    expert_hidden = [m for m in expert.modules()
                     if isinstance(m, linear_types) and m is not out_layer_new]
    parent_hidden = [m for m in parent_model.modules()
                     if isinstance(m, linear_types) and m is not out_layer_par]

    def _shapes(mods):
        return [tuple(m.weight.shape) for m in mods]

    # Align hidden layers from the output end (reversed) so output-adjacent
    # layers pair up; every expert layer must find a matching parent layer.
    n_hidden_copied = 0
    for mod_new, mod_par in zip(reversed(expert_hidden), reversed(parent_hidden)):
        if _copy_module(mod_new, mod_par):
            n_hidden_copied += 1

    output_copied = False
    if copy_output:
        output_copied = _copy_module(out_layer_new, out_layer_par)

    hidden_ok = n_hidden_copied == len(expert_hidden)
    output_ok = output_copied or not copy_output
    if not (hidden_ok and output_ok):
        msg = (
            f"[Init] ARCHITECTURE MISMATCH — parent_weights copy failed: "
            f"copied {n_hidden_copied}/{len(expert_hidden)} hidden layers, "
            f"output copied={output_copied} (copy_output={copy_output}). "
            f"Expert hidden shapes {_shapes(expert_hidden)} + output "
            f"{tuple(out_layer_new.weight.shape)} vs parent hidden "
            f"{_shapes(parent_hidden)} + output {tuple(out_layer_par.weight.shape)}. "
            f"With init.hidden=parent_weights the expert architecture must match "
            f"its parent's; set experts_architecture accordingly or use a "
            f"different init.hidden."
        )
        logger.error("!" * 70)
        logger.error(msg)
        logger.error("!" * 70)
        raise RuntimeError(msg)

    if not output_copied:
        with torch.no_grad():
            nn.init.zeros_(out_layer_new.weight)
            if out_layer_new.bias is not None:
                nn.init.zeros_(out_layer_new.bias)

    if output_copied:
        logger.info(f"  [Init] Copied {n_hidden_copied} hidden layers from parent; output copied")
    else:
        logger.info(f"  [Init] Copied {n_hidden_copied} hidden layers from parent; output zeroed")
