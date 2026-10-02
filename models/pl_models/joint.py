"""Joint space-group classification and CrystalGNN occupation flow."""

import hydra
import torch
import torch.nn.functional as F

from .base import resolve_config
from .composition_encoder import CompositionFeatures, CrystalCompositionEncoder
from .count_conserving import formula_space_group_mask
from .flow import DiscreteFlowModule
from .spg_predictor import SpaceGroupHead


class JointSpaceGroupWyckoffModule(DiscreteFlowModule):
    """Share static chemical features while retaining the conditional Wyckoff flow."""

    def __init__(
        self,
        optimizer_config,
        decoder,
        sg_head,
        num_elements,
        max_num_atoms,
        flow_source,
        sg_loss_weight=1.0,
        flow_loss_weight=1.0,
        zero_df_loss_weight=1.0,
        inf_df_loss_weight=1.0,
        conditional_composition=True,
        validation_seed=42,
        decay_matrix_weights_only=False,
        flow_encoder_grad_scale=1.0,
        label_smoothing=0.0,
    ):
        decoder = {**resolve_config(decoder), "external_composition": True}
        super().__init__(
            optimizer_config=optimizer_config,
            decoder=decoder,
            num_elements=num_elements,
            max_num_atoms=max_num_atoms,
            flow_source=flow_source,
            zero_df_loss_weight=zero_df_loss_weight,
            inf_df_loss_weight=inf_df_loss_weight,
            conditional_composition=conditional_composition,
            validation_seed=validation_seed,
            label_smoothing=label_smoothing,
        )
        sg_head = resolve_config(sg_head)
        self.task = "joint"
        self.save_hyperparameters(
            {
                "task": self.task,
                "sg_head": sg_head,
                "sg_loss_weight": sg_loss_weight,
                "flow_loss_weight": flow_loss_weight,
                "decay_matrix_weights_only": decay_matrix_weights_only,
                "flow_encoder_grad_scale": flow_encoder_grad_scale,
            }
        )
        if min(sg_loss_weight, flow_loss_weight) <= 0:
            raise ValueError("joint loss weights must be positive")
        self.sg_loss_weight = sg_loss_weight
        self.flow_loss_weight = flow_loss_weight
        self.decay_matrix_weights_only = decay_matrix_weights_only
        if not 0 <= flow_encoder_grad_scale <= 1:
            raise ValueError("flow_encoder_grad_scale must be between 0 and 1")
        self.flow_encoder_grad_scale = float(flow_encoder_grad_scale)
        self.composition_encoder = CrystalCompositionEncoder(
            num_elements, self.decoder.element_dim, self.decoder.hidden_dim
        )
        self.sg_head = SpaceGroupHead(sg_head, self.decoder.hidden_dim)

    def configure_optimizers(self):
        if not self.decay_matrix_weights_only:
            return super().configure_optimizers()
        # Apply decay to matrices in every branch, excluding biases, norm
        # scales, and other one-dimensional parameters.
        groups = [
            {
                "params": [
                    parameter for parameter in self.parameters() if parameter.ndim >= 2
                ],
            },
            {
                "params": [
                    parameter for parameter in self.parameters() if parameter.ndim < 2
                ],
                "weight_decay": 0.0,
            },
        ]
        return hydra.utils.instantiate(
            self.optimizer_config, params=groups, _convert_="all"
        )

    def encode_composition(self, composition):
        return self.composition_encoder(composition)

    def flow_loss(self, batch, composition_features=None):
        if composition_features is None:
            composition_features = self.encode_composition(batch.composition)
        if self.flow_encoder_grad_scale != 1:
            # Preserve feature values while scaling only the Flow gradient into
            # the shared encoder. SG and decoder parameter gradients are intact.
            composition_features = CompositionFeatures(
                *(
                    value.detach()
                    + self.flow_encoder_grad_scale * (value - value.detach())
                    for value in composition_features
                )
            )
        return super().flow_loss(batch, composition_features)

    def space_group_logits(self, composition, features=None):
        if features is None:
            features = self.encode_composition(composition)
        logits = self.sg_head(features.pooled)
        feasible = formula_space_group_mask(composition, self.max_num_atoms)
        return logits.masked_fill(~feasible, float("-inf"))

    @torch.inference_mode()
    def predict_space_groups(self, composition):
        """Return probabilities for indices 0..230; index zero is always masked."""
        composition = composition.to(self.device)
        return self.space_group_logits(composition).softmax(dim=-1)

    def forward(self, batch):
        features = self.encode_composition(batch.composition)
        logits = self.space_group_logits(batch.composition, features)
        target = batch.space_group.long().reshape(-1)
        sg_loss = F.cross_entropy(logits, target)
        flow = self.flow_loss(batch, features)
        top_five = logits.topk(5, dim=-1).indices
        return {
            "loss": self.sg_loss_weight * sg_loss
            + self.flow_loss_weight * flow["loss"],
            "sg_loss": sg_loss,
            "flow_loss": flow["loss"],
            "zero_df_loss": flow["zero_df_loss"],
            "inf_df_loss": flow["inf_df_loss"],
            "spg_top1": (top_five[:, 0] == target).float().mean(),
            "spg_top5": (top_five == target[:, None]).any(dim=-1).float().mean(),
        }
