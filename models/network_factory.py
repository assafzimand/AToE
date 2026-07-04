"""Factory function for creating network models (FCNet, ResNetModel, or PirateNet)."""

from typing import List, Dict


def create_network(layers: List[int], activation: str, config: Dict,
                   is_base: bool = True, expert_type: str = 'mlp'):
    """Create a network model based on expert_type.

    Args:
        layers: Architecture list [input_dim, hidden..., output_dim].
        activation: Activation function name.
        config: Full configuration dictionary.
        is_base: Whether this is a base model (validates input_dim).
        expert_type: 'mlp' | 'resnet' | 'piratenet'.

    Returns:
        An nn.Module with the standard PINN interface
        (forward, get_activation_dim, get_layer_names, etc.).
    """
    if expert_type == 'resnet':
        from models.resnet_model import ResNetModel
        return ResNetModel(layers, activation, config, is_base)
    if expert_type == 'piratenet':
        from models.pirate_net import PirateNet
        return PirateNet(layers, activation, config, is_base)
    from models.fc_model import FCNet
    return FCNet(layers, activation, config, is_base)
