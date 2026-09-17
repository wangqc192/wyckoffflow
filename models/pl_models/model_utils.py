"""Graph construction helpers for Wyckoff models."""

import torch
from torch_geometric.data import Data

from ..common import lookup_tables


class WyckoffData(Data):
    """Wyckoff graph with global metadata labels that PyG must not offset."""

    def __inc__(self, key, value, *args, **kwargs):
        if key in {"target_index", "sampling_group"}:
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def create_x_matrix(x_inf_dof, x_0_dof, zero_dof):
    x = torch.zeros(
        (x_inf_dof.shape[0] + x_0_dof.shape[0], x_inf_dof.shape[1] + 1),
        device=x_inf_dof.device,
    )
    x[zero_dof, 0] = x_0_dof.float()
    x[~zero_dof, 1:] = x_inf_dof.float()
    return x


def get_degrees_of_freedom(space_group, device=None):
    values = lookup_tables.spg_wyckoff_degrees_of_freedom[str(int(space_group))]
    return torch.tensor(list(reversed(values.values())), device=device)


def create_wyckoff_graph(space_group, x_0_dof, x_inf_dof):
    """Create a fully connected graph for one space group."""

    space_group = int(space_group)
    device = x_inf_dof.device
    degrees_of_freedom = get_degrees_of_freedom(space_group, device)
    zero_dof = degrees_of_freedom == 0
    num_pos = degrees_of_freedom.numel()
    positions = torch.arange(num_pos, device=device)
    multiplicity_values = lookup_tables.spg_wyckoff_multiplicities[str(space_group)]
    multiplicities = torch.tensor(
        list(reversed(multiplicity_values.values())), device=device
    )

    return WyckoffData(
        x=create_x_matrix(x_inf_dof, x_0_dof, zero_dof),
        edge_index=torch.stack(
            [positions.repeat_interleave(num_pos), positions.repeat(num_pos)]
        ),
        space_group=torch.tensor(space_group, device=device),
        x_0_dof=x_0_dof,
        x_inf_dof=x_inf_dof,
        zero_dof=zero_dof,
        degrees_of_freedom=degrees_of_freedom,
        multiplicities=multiplicities,
        wyckoff_pos_idx=positions,
        num_pos=torch.tensor([num_pos], device=device),
        num_nodes=num_pos,
        num_0_dof=zero_dof.sum(),
        num_inf_dof=(~zero_dof).sum(),
    )
