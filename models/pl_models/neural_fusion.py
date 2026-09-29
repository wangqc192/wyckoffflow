"""Neural space-group fusion trained with out-of-fold expert predictions."""

import torch
from torch import nn

from models.pl_models.chemical_sg import (
    ChemicalCompositionFeatures,
    ChemicalSpaceGroupEnsemble,
    ResidualBlock,
)
from models.pl_models.count_conserving import formula_space_group_mask


class NeuralFusion(nn.Module):
    def __init__(
        self,
        experts,
        width=128,
        dropout=0.35,
        correction_limit=3.0,
        num_elements=100,
        base_weights=None,
    ):
        super().__init__()
        self.features = ChemicalCompositionFeatures(
            num_elements=num_elements, count_histogram=True
        )
        self.context = nn.Sequential(
            nn.Linear(self.features.output_dim + 231, width),
            nn.LayerNorm(width),
            nn.GELU(),
            ResidualBlock(width, dropout),
            nn.Linear(width, 64),
        )
        self.group_embedding = nn.Embedding(231, 16)
        self.score = nn.Sequential(
            nn.Linear(2 * experts + 2 + 64 + 16, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        self.correction_limit = correction_limit
        self.register_buffer(
            "base_weights",
            (
                None
                if base_weights is None
                else torch.tensor(base_weights, dtype=torch.float32)
            ),
        )

    def base_probability(self, probability):
        if self.base_weights is None:
            return probability.mean(1)
        return (probability * self.base_weights[None, :, None]).sum(1)

    def forward(self, composition, feasible, probability, features=None):
        if features is None:
            features = self.features(composition)
        context = self.context(torch.cat((features, feasible.float()), -1))
        mean = self.base_probability(probability)
        logp = probability.clamp_min(1e-8).log().transpose(1, 2)
        mean_logp = mean.clamp_min(1e-8).log()
        signals = torch.cat(
            (
                logp / 5,
                (logp - mean_logp[..., None]).clamp(-8, 8) / 5,
                mean_logp[..., None] / 5,
                mean[..., None],
                context[:, None, :].expand(-1, 231, -1),
                self.group_embedding.weight[None].expand(len(mean), -1, -1),
            ),
            -1,
        )
        correction = self.correction_limit * self.score(signals).squeeze(-1).tanh()
        logits = mean.clamp_min(1e-30).log() + correction
        return logits.masked_fill(~feasible, -torch.inf), correction


class NeuralSpaceGroupFusion(nn.Module):
    """Self-contained composition-only inference with learned expert fusion."""

    def __init__(self, checkpoint):
        super().__init__()
        self.groups = nn.ModuleList(
            ChemicalSpaceGroupEnsemble(members)
            for members in checkpoint["expert_groups"]
        )
        self.num_elements = self.groups[0].num_elements
        self.max_num_atoms = self.groups[0].max_num_atoms
        self.fusion = NeuralFusion(**checkpoint["fusion_config"])
        self.fusion.load_state_dict(checkpoint["fusion_state_dict"], strict=True)
        prototype_head = checkpoint.get("prototype_head")
        self.prototype_member = None
        self.prototype_weight = 0.0
        if prototype_head is not None:
            self.prototype_member = prototype_head["member"]
            self.prototype_weight = prototype_head["weight"]
            self.register_buffer(
                "prototype_space_groups",
                torch.tensor(prototype_head["space_group_ids"], dtype=torch.long),
            )

    @torch.inference_mode()
    def predict_space_groups(self, composition):
        return self.predict_space_groups_with_base(composition)[0]

    @torch.inference_mode()
    def predict_space_groups_with_base(self, composition):
        """Return learned probabilities and the underlying neural proposal mixture."""
        composition = composition.to(next(self.parameters()).device)
        feasible = formula_space_group_mask(composition, self.max_num_atoms)
        features = self.fusion.features(composition)
        probabilities = []
        base_probabilities = []
        for group in self.groups:
            expert_probabilities = []
            prototype_probability = None
            for index, member in enumerate(group.members):
                if index == self.prototype_member:
                    expert_logits, prototype_logits = member(
                        composition, feasible, return_auxiliary=True
                    )
                    prototype_logits = prototype_logits.float().masked_fill(
                        ~feasible[:, self.prototype_space_groups], -torch.inf
                    )
                    prototype_probability = torch.zeros_like(
                        expert_logits, dtype=torch.float32
                    )
                    prototype_probability.scatter_add_(
                        1,
                        self.prototype_space_groups[None].expand(len(composition), -1),
                        prototype_logits.softmax(-1),
                    )
                else:
                    expert_logits = member(composition, feasible)
                expert_probabilities.append(expert_logits.softmax(-1))
            expert_probabilities = torch.stack(expert_probabilities, dim=1)
            logits, _ = self.fusion(
                composition, feasible, expert_probabilities, features
            )
            probability = logits.softmax(-1)
            if prototype_probability is not None:
                probability = (
                    1 - self.prototype_weight
                ) * probability + self.prototype_weight * prototype_probability
            probabilities.append(probability)
            base_probabilities.append(expert_probabilities.mean(1))
        return torch.stack(probabilities).mean(0), torch.stack(base_probabilities).mean(
            0
        )
