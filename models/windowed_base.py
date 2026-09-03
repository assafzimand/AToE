"""Frozen stitched time-window root used as ``model.base_model``.

A ``WindowedBase`` holds one base network per time window (the collected
``window_{i}_best_model_<segment>.pt`` files of a time-marching run) and
routes each input row to the window owning its t.

Seam convention: at t exactly on the window-i / window-i+1 boundary the
prediction comes from window i — the window that actually trained up to that
endpoint (window i+1 only ever saw window i's prediction there, as its IC).

The wrapper exposes the surface the framework touches on ``model.base_model``:
callable forward, ``parameters()``, ``.layers`` / ``get_layer_names()`` /
``get_activation_dim()`` (proxied to window 0 — all windows share one
architecture), and a standard ``state_dict()`` (keys ``windows.{i}.*`` plus
the ``window_t_ends`` buffer) so adaptive checkpoints round-trip through
``state_dict_extended`` / ``load_state_dict_extended``.
"""
from typing import Dict, List

import torch
import torch.nn as nn

from models.network_factory import create_network


class WindowedBase(nn.Module):
    def __init__(self, nets: List[nn.Module], t_ends: List[float]):
        super().__init__()
        if len(nets) != len(t_ends):
            raise ValueError(
                f"WindowedBase: {len(nets)} nets but {len(t_ends)} t_ends")
        if list(t_ends) != sorted(t_ends):
            raise ValueError(f"WindowedBase: t_ends not ascending: {t_ends}")
        archs = {tuple(n.layers) for n in nets}
        if len(archs) != 1:
            raise ValueError(
                f"WindowedBase: windows disagree on architecture: {archs}")
        self.windows = nn.ModuleList(nets)
        # float64 buffer: seam comparisons happen in the input dtype after a
        # cast in forward, so storing full precision loses nothing.
        self.register_buffer(
            'window_t_ends', torch.tensor(t_ends, dtype=torch.float64))
        self.layers = list(nets[0].layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[0] == 0:
            return inputs.new_zeros((0, self.layers[-1]))
        t = inputs[:, -1].detach().contiguous()
        ends = self.window_t_ends.to(dtype=t.dtype, device=t.device)
        # First window whose t_end >= t: window i owns its right endpoint
        # (searchsorted side='left'); clamp catches numerical t > t_max.
        idx = torch.searchsorted(ends, t).clamp_(max=len(self.windows) - 1)
        out = None
        for i, net in enumerate(self.windows):
            m = idx == i
            if not bool(m.any()):
                continue
            o = net(inputs[m])
            if out is None:
                out = torch.zeros(inputs.shape[0], o.shape[-1],
                                  dtype=o.dtype, device=o.device)
            out[m] = o
        return out

    def get_layer_names(self) -> List[str]:
        return self.windows[0].get_layer_names()

    def get_activation_dim(self) -> int:
        return self.windows[0].get_activation_dim()

    @classmethod
    def from_state_dict(cls, state_dict: Dict, architecture: List[int],
                        activation: str, config: Dict,
                        expert_type: str = 'mlp') -> 'WindowedBase':
        """Rebuild a WindowedBase from its own ``state_dict()`` (the
        ``base_model`` entry of an adaptive checkpoint saved by a run whose
        base was windowed)."""
        idxs = sorted({int(k.split('.')[1]) for k in state_dict
                       if k.startswith('windows.')})
        if idxs != list(range(len(idxs))) or not idxs:
            raise ValueError(
                f"WindowedBase.from_state_dict: bad window indices {idxs}")
        nets = []
        for i in idxs:
            prefix = f'windows.{i}.'
            sub = {k[len(prefix):]: v for k, v in state_dict.items()
                   if k.startswith(prefix)}
            net = create_network(list(architecture), activation, config,
                                 is_base=True, expert_type=expert_type)
            net.load_state_dict(sub)
            nets.append(net)
        t_ends = state_dict['window_t_ends'].tolist()
        return cls(nets, t_ends)

    def __repr__(self) -> str:
        ends = [round(float(e), 6) for e in self.window_t_ends]
        return (f"WindowedBase({len(self.windows)} windows, t_ends={ends}, "
                f"arch={self.layers})")
