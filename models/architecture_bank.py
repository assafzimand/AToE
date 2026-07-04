"""Architecture bank for AToE adaptive expert sizing.

Maps approximate parameter capacity to hidden layer configurations.
Entries span 1k-20k parameters (reference: input_dim=2, output_dim=1)
with sub-monotonic growth in both width and depth (2-7 hidden layers).
Width may dip at depth transitions but is non-decreasing within each
depth group.

Parameter counts for each bank entry (input_dim=2, output_dim=1, fourier_dim=64):
  - MLP:        standard fully-connected
  - MLP+RWF:    MLP with Random Weight Factorization on all hidden layers
  - ResNet:     residual network (pairs of hidden layers form ResBlocks)
  - ResNet+RWF: ResNet with RWF on input_proj + all ResBlock layers + leftover
  - PirateNet:  UV-gated residual net with Fourier features (ff_out=128)
  - Pirate+RWF: PirateNet with RWF on input_proj, U/V projections, hidden layers

  RWF overhead = n_hidden_layers × h  (one scale scalar per neuron per hidden layer)
  PirateNet formula: 3×(128×h + h) + n_hidden×(h²+h) + h×1 + 1  [+ RWF overhead]

  MLP label  ->     MLP  MLP+RWF   ResNet  ResNet+RWF  PirateNet  Pirate+RWF  Architecture
  ──────────────────────────────────────────────────────────────────────────────────────────
#    1000  ->     987     1045    1857        1944      12993       13138    [29, 29]
#    1500  ->    1477     1549    2809        2917      16633       16813    [36, 36]
#    2000  ->    1981     2071    2911        3031      14431       14611    [30, 30, 30]
#    2500  ->    2517     2619    3707        3843      16763       16967    [34, 34, 34]
#    3000  ->    2961     3072    4367        4515      18575       18797    [37, 37, 37]
#    3500  ->    3441     3561    5081        5241      20441       20681    [40, 40, 40]
#    4000  ->    3921     4061    5181        5356      18621       18866    [35, 35, 35, 35]
#    4500  ->    4599     4751    6081        6271      20673       20939    [38, 38, 38, 38]
#    5000  ->    5081     5241    6721        6921      22081       22361    [40, 40, 40, 40]
#    5500  ->    5587     5755    7393        7603      23521       23815    [42, 42, 42, 42]
#    6000  ->    6081     6271    7563        7791      22155       22459    [38, 38, 38, 38, 38]
#    6500  ->    6397     6592    7957        8191      22933       23245    [39, 39, 39, 39, 39]
#    7000  ->    7053     7258    8775        9021      24519       24847    [41, 41, 41, 41, 41]
#    7500  ->    7393     7603    9199        9451      25327       25663    [42, 42, 42, 42, 42]
#    8000  ->    8097     8317   10077       10341      26973       27325    [44, 44, 44, 44, 44]
#    8500  ->    8361     8601   10001       10281      25361       25721    [40, 40, 40, 40, 40, 40]
#    9000  ->    9199     9451   11005       11299      27133       27511    [42, 42, 42, 42, 42, 42]
#    9500  ->    9633     9891   11525       11826      28037       28424    [43, 43, 43, 43, 43, 43]
#   10000  ->   10077    10341   12057       12365      28953       29349    [44, 44, 44, 44, 44, 44]
#   10500  ->   10531    10801   12601       12916      29881       30286    [45, 45, 45, 45, 45, 45]
#   11000  ->   10995    11271   13157       13479      30821       31235    [46, 46, 46, 46, 46, 46]
#   11500  ->   11469    11751   13725       14054      31773       32196    [47, 47, 47, 47, 47, 47]
#   12000  ->   11953    12241   14305       14641      32737       33169    [48, 48, 48, 48, 48, 48]
#   12500  ->   12447    12741   14897       15240      33713       34154    [49, 49, 49, 49, 49, 49]
#   13000  ->   12951    13251   15501       15851      34701       35151    [50, 50, 50, 50, 50, 50]
#   13500  ->   13465    13771   16117       16474      35701       36160    [51, 51, 51, 51, 51, 51]
#   14000  ->   13989    14301   16745       17109      36713       37181    [52, 52, 52, 52, 52, 52]
#   14500  ->   14523    14841   17385       17756      37737       38214    [53, 53, 53, 53, 53, 53]
#   15000  ->   15067    15391   18037       18415      38773       39259    [54, 54, 54, 54, 54, 54]
#   15500  ->   15621    15951   18701       19086      39821       40316    [55, 55, 55, 55, 55, 55]
#   16000  ->   16185    16521   19377       19769      40881       41385    [56, 56, 56, 56, 56, 56]
#   16500  ->   16759    17101   20065       20464      41953       42466    [57, 57, 57, 57, 57, 57]
#   17000  ->   17343    17691   20765       21171      43037       43559    [58, 58, 58, 58, 58, 58]
#   17500  ->   17937    18291   21477       21890      44133       44664    [59, 59, 59, 59, 59, 59]
#   18000  ->   18037    18415   21007       21439      41743       42283    [54, 54, 54, 54, 54, 54, 54]
#   18500  ->   18701    19086   21781       22221      42901       43451    [55, 55, 55, 55, 55, 55, 55]
#   19000  ->   19377    19769   22569       23017      44073       44633    [56, 56, 56, 56, 56, 56, 56]
#   19500  ->   19501    19917   22257       22725      42225       42797    [52, 52, 52, 52, 52, 52, 52, 52]
#   20000  ->   20065    20464   23371       23827      45259       45829    [57, 57, 57, 57, 57, 57, 57]
"""

from collections import OrderedDict
from typing import List


ARCHITECTURE_BANK = OrderedDict([
    # 2 hidden layers
    (1000,  [29, 29]),
    (1500,  [36, 36]),
    # 3 hidden layers
    (2000,  [30, 30, 30]),
    (2500,  [34, 34, 34]),
    (3000,  [37, 37, 37]),
    (3500,  [40, 40, 40]),
    # 4 hidden layers
    (4000,  [35, 35, 35, 35]),
    (4500,  [38, 38, 38, 38]),
    (5000,  [40, 40, 40, 40]),
    (5500,  [42, 42, 42, 42]),
    # 5 hidden layers
    (6000,  [38, 38, 38, 38, 38]),
    (6500,  [39, 39, 39, 39, 39]),
    (7000,  [41, 41, 41, 41, 41]),
    (7500,  [42, 42, 42, 42, 42]),
    (8000,  [44, 44, 44, 44, 44]),
    # 6 hidden layers
    (8500,  [40, 40, 40, 40, 40, 40]),
    (9000,  [42, 42, 42, 42, 42, 42]),
    (9500,  [43, 43, 43, 43, 43, 43]),
    (10000, [44, 44, 44, 44, 44, 44]),
    (10500, [45, 45, 45, 45, 45, 45]),
    (11000, [46, 46, 46, 46, 46, 46]),
    (11500, [47, 47, 47, 47, 47, 47]),
    (12000, [48, 48, 48, 48, 48, 48]),
    (12500, [49, 49, 49, 49, 49, 49]),
    (13000, [50, 50, 50, 50, 50, 50]),
    (13500, [51, 51, 51, 51, 51, 51]),
    (14000, [52, 52, 52, 52, 52, 52]),
    (14500, [53, 53, 53, 53, 53, 53]),
    (15000, [54, 54, 54, 54, 54, 54]),
    (15500, [55, 55, 55, 55, 55, 55]),
    (16000, [56, 56, 56, 56, 56, 56]),
    (16500, [57, 57, 57, 57, 57, 57]),
    (17000, [58, 58, 58, 58, 58, 58]),
    (17500, [59, 59, 59, 59, 59, 59]),
    # 7 hidden layers
    (18000, [54, 54, 54, 54, 54, 54, 54]),
    (18500, [55, 55, 55, 55, 55, 55, 55]),
    (19000, [56, 56, 56, 56, 56, 56, 56]),
    (19500, [52, 52, 52, 52, 52, 52, 52, 52]),
    (20000, [57, 57, 57, 57, 57, 57, 57]),
])


def compute_param_count(
    hidden_layers: List[int], input_dim: int, output_dim: int,
    expert_type: str = 'mlp',
) -> int:
    """Compute total parameter count for a network.

    For 'mlp': standard fully-connected count (weights + biases per layer).
    For 'resnet': mirrors ResNetModel — input_proj + ResBlocks (pairs of
    hidden layers) + optional leftover plain layer + output_proj.
    All hidden widths must be identical for resnet.

    Args:
        hidden_layers: List of hidden layer widths (no input/output dims).
        input_dim: Input dimension of the network.
        output_dim: Output dimension of the network.
        expert_type: 'mlp' or 'resnet'.

    Returns:
        Total number of trainable parameters (weights + biases).
    """
    if expert_type == 'resnet':
        h = hidden_layers[0]
        n_hidden = len(hidden_layers)
        # input_proj: Linear(input_dim, h)
        total = input_dim * h + h
        # ResBlocks: each block has 2 × Linear(h, h)
        n_blocks = n_hidden // 2
        total += n_blocks * 2 * (h * h + h)
        # leftover plain layer: Linear(h, h) if odd number of hidden layers
        if n_hidden % 2 == 1:
            total += h * h + h
        # output_proj: Linear(h, output_dim)
        total += h * output_dim + output_dim
        return total

    # MLP
    layers = [input_dim] + list(hidden_layers) + [output_dim]
    total = 0
    for i in range(len(layers) - 1):
        total += layers[i] * layers[i + 1] + layers[i + 1]
    return total


def get_architecture_for_capacity(
    target_capacity: float, input_dim: int, output_dim: int,
    expert_type: str = 'mlp',
) -> List[int]:
    """Look up the bank entry whose actual param count is closest to target.

    Computes exact parameter counts (respecting expert_type) using the
    provided input/output dims, then returns the full architecture
    [input_dim, *hidden, output_dim] for the best-matching entry.
    Capacities above the bank maximum return the largest entry.

    Args:
        target_capacity: Desired number of parameters.
        input_dim: Problem input dimension (spatial_dim + 1).
        output_dim: Problem output dimension.
        expert_type: 'mlp' or 'resnet' — affects how params are counted.

    Returns:
        Full architecture list [input_dim, h1, h2, ..., output_dim].
    """
    best_hidden = None
    best_diff = float('inf')

    for _label, hidden_layers in ARCHITECTURE_BANK.items():
        actual = compute_param_count(
            hidden_layers, input_dim, output_dim, expert_type)
        diff = abs(actual - target_capacity)
        if diff < best_diff:
            best_diff = diff
            best_hidden = hidden_layers

    return [input_dim] + list(best_hidden) + [output_dim]
