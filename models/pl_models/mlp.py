"""MLP building blocks shared by the model modules."""

import torch.nn as nn


class MLP(nn.Sequential):
    """Sequential MLP with fixed layer indices for every dropout probability."""

    _version = 2

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Older get_mlp omitted dropout layers in the SG and WyckoffGNN MLPs.
        # Translate that layout here so callers only handle probabilities.
        if local_metadata.get("version", 1) < 2:
            legacy_keys = {}
            legacy_index = 0
            for name, layer in self.named_children():
                if isinstance(layer, nn.Dropout):
                    continue
                for key in layer.state_dict():
                    legacy_keys[f"{prefix}{legacy_index}.{key}"] = (
                        f"{prefix}{name}.{key}"
                    )
                legacy_index += 1
            saved_keys = {key for key in state_dict if key.startswith(prefix)}
            if saved_keys == legacy_keys.keys():
                renamed = {new: state_dict.pop(old) for old, new in legacy_keys.items()}
                state_dict.update(renamed)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def get_mlp(
    input_dim,
    output_dim,
    hidden_dim,
    num_hidden_layers=1,
    activation="SiLU",
    *,
    layer_norm=False,
    dropout=0.0,
):
    """Build exactly ``num_hidden_layers`` hidden blocks and a linear output.

    Each hidden block is Linear -> optional LayerNorm -> activation -> Dropout.
    Zero disables dropout without changing parameter keys. The output is linear.
    """
    activation_cls = getattr(nn, activation)
    layers = []
    current_dim = input_dim
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(current_dim, hidden_dim))
        current_dim = hidden_dim
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation_cls())
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(current_dim, output_dim))
    return MLP(*layers)
