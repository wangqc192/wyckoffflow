"""Composition-only space-group prediction with fixed chemical descriptors."""

import json
from pathlib import Path

import torch
from torch import nn

from models.common.lookup_tables import chemical_symbols
from models.pl_models.count_conserving import formula_space_group_mask


def chemical_descriptors(num_elements):
    from aviary import PKG_DIR

    tables = []
    for name in ("matscholar200", "cgcnn92"):
        entries = json.loads(
            (Path(PKG_DIR) / f"embeddings/element/{name}.json").read_text()
        )
        values = torch.tensor(
            [entries[s] for s in chemical_symbols[1 : num_elements + 1]],
            dtype=torch.float32,
        )
        values = (values - values.mean(0)) / values.std(0).clamp_min(1e-6)
        tables.append(values)
    return torch.cat(tables, dim=-1)


def masked_classification_loss(logits, target, feasible, smoothing=0.0):
    """Smooth only over feasible classes, avoiding zero times negative infinity."""
    log_prob = logits.masked_fill(~feasible, -torch.inf).log_softmax(-1)
    nll = -log_prob.gather(1, target[:, None]).squeeze(1)
    smooth = -log_prob.masked_fill(~feasible, 0).sum(-1) / feasible.sum(-1)
    return ((1 - smoothing) * nll + smoothing * smooth).mean()


def atom_count_histogram(composition):
    """Encode the counts in this input formula; contains no dataset statistics."""
    counts = composition[:, 1:]
    bins = torch.arange(1, 55, device=counts.device)
    return (counts[..., None] == bins).sum(1).float().log1p()


class ChemicalCompositionFeatures(nn.Module):
    def __init__(self, num_elements=100, role_groups=0, count_histogram=False):
        super().__init__()
        self.register_buffer("descriptors", chemical_descriptors(num_elements))
        self.role_groups = role_groups
        self.count_histogram = count_histogram
        self.output_dim = (
            4 * self.descriptors.shape[-1]
            + 3 * num_elements
            + 2
            + role_groups * (self.descriptors.shape[-1] + 2)
            + (54 if count_histogram else 0)
        )

    def forward(self, composition):
        counts = composition[:, 1:].float()
        present = counts > 0
        total = counts.sum(-1, keepdim=True)
        species = present.sum(-1, keepdim=True)
        fractions = counts / total
        descriptors = self.descriptors.unsqueeze(0).expand(len(counts), -1, -1)
        means = present.float() @ self.descriptors / species
        weighted = fractions @ self.descriptors
        largest = descriptors.masked_fill(~present[..., None], -torch.inf).amax(1)
        smallest = descriptors.masked_fill(~present[..., None], torch.inf).amin(1)
        parts = [
            means,
            weighted,
            largest,
            smallest,
            counts.log1p(),
            fractions,
            counts.sort(dim=-1, descending=True).values.log1p(),
            total.log1p(),
            species.float().log1p(),
        ]
        remaining = counts.clone()
        if self.count_histogram:
            parts.append(atom_count_histogram(composition))
        for _ in range(self.role_groups):
            count = remaining.amax(-1, keepdim=True)
            members = (counts == count) & present
            size = members.sum(-1, keepdim=True)
            chemistry = members.float() @ self.descriptors / size.clamp_min(1)
            parts.extend((chemistry, count.log1p(), size.float().log1p()))
            remaining = remaining.masked_fill(members, 0)
        return torch.cat(parts, dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.Dropout(dropout),
        )

    def forward(self, value):
        return value + self.network(value)


class ChemicalSpaceGroupPredictor(nn.Module):
    """Featurize elemental chemistry, atom counts and optional element interactions."""

    def __init__(
        self,
        num_elements=100,
        width=384,
        layers=3,
        dropout=0.15,
        attention=False,
        role_groups=0,
        conditioned=False,
        use_feasibility=True,
        auxiliary_classes=0,
        count_histogram=False,
        count_embedding=False,
        prototype_space_groups=None,
        prototype_weight=0.0,
    ):
        super().__init__()
        self.features = ChemicalCompositionFeatures(
            num_elements, role_groups=role_groups, count_histogram=count_histogram
        )
        self.attention = attention
        self.use_count_embedding = count_embedding
        self.conditioned = conditioned
        self.use_feasibility = use_feasibility
        self.prototype_weight = prototype_weight
        if prototype_weight:
            self.register_buffer(
                "prototype_space_groups",
                torch.tensor(prototype_space_groups, dtype=torch.long),
            )
        self.project = nn.Sequential(
            nn.Linear(self.features.output_dim + 231, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        if attention:
            token_dim = 192
            self.element_embedding = nn.Embedding(num_elements + 1, token_dim)
            if count_embedding:
                self.count_embedding = nn.Embedding(129, token_dim, padding_idx=0)
            self.token_project = nn.Linear(
                self.features.descriptors.shape[1] + 2, token_dim
            )
            layer = nn.TransformerEncoderLayer(
                token_dim,
                6,
                2 * token_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, 3, enable_nested_tensor=False
            )
            self.interaction_project = nn.Linear(2 * token_dim, width)
        self.blocks = nn.Sequential(
            *[ResidualBlock(width, dropout) for _ in range(layers)]
        )
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 231))
        if auxiliary_classes:
            self.auxiliary = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, auxiliary_classes)
            )
        if conditioned:
            self.stoichiometry = nn.Sequential(
                nn.Linear(num_elements + 231 + (54 if count_histogram else 0), 128),
                nn.GELU(),
                ResidualBlock(128, dropout),
                nn.LayerNorm(128),
            )
            self.modulation = nn.ModuleList(
                nn.Linear(128, 2 * width) for _ in range(layers)
            )
            self.stoichiometry_output = nn.Linear(128, 231)

    def forward(self, composition, feasible, features=None, return_auxiliary=False):
        if not self.use_feasibility:
            feasible = torch.ones_like(feasible)
            feasible[:, 0] = False
        if features is None:
            features = self.features(composition)
        hidden = self.project(torch.cat((features, feasible.float()), dim=-1))
        if self.attention:
            counts, indices = composition[:, 1:].float().sort(dim=-1, descending=True)
            length = int((counts > 0).sum(-1).max())
            counts, indices = counts[:, :length], indices[:, :length]
            present = counts > 0
            fractions = counts / counts.sum(-1, keepdim=True)
            descriptors = self.features.descriptors[indices]
            tokens = self.element_embedding(indices + 1) + self.token_project(
                torch.cat(
                    (descriptors, counts.log1p()[..., None], fractions[..., None]),
                    dim=-1,
                )
            )
            if self.use_count_embedding:
                tokens = tokens + self.count_embedding(counts.long())
            tokens = self.transformer(tokens, src_key_padding_mask=~present)
            mean = (tokens * present[..., None]).sum(1) / present.sum(-1, keepdim=True)
            weighted = (tokens * fractions[..., None]).sum(1)
            hidden = hidden + self.interaction_project(
                torch.cat((mean, weighted), dim=-1)
            )
        if self.conditioned:
            counts = composition[:, 1:].float().sort(-1, descending=True).values.log1p()
            if self.features.count_histogram:
                counts = torch.cat((counts, atom_count_histogram(composition)), -1)
            stoichiometry = self.stoichiometry(
                torch.cat((counts, feasible.float()), -1)
            )
            for block, modulation in zip(self.blocks, self.modulation):
                scale, shift = modulation(stoichiometry).chunk(2, -1)
                hidden = block(hidden * (1 + scale.tanh()) + shift)
            logits = self.output(hidden) + self.stoichiometry_output(stoichiometry)
        else:
            hidden = self.blocks(hidden)
            logits = self.output(hidden)
        logits = logits.masked_fill(~feasible, -torch.inf)
        auxiliary_logits = None
        if return_auxiliary or self.prototype_weight:
            auxiliary_logits = self.auxiliary(hidden)
        if self.prototype_weight:
            class_probabilities = (
                auxiliary_logits.float()
                .masked_fill(~feasible[:, self.prototype_space_groups], -torch.inf)
                .softmax(-1)
            )
            prototype_probability = torch.zeros_like(logits, dtype=torch.float32)
            prototype_probability.scatter_add_(
                1,
                self.prototype_space_groups[None].expand(len(composition), -1),
                class_probabilities,
            )
            probability = (1 - self.prototype_weight) * logits.float().softmax(
                -1
            ) + self.prototype_weight * prototype_probability
            logits = (
                probability.clamp_min(1e-30).log().masked_fill(~feasible, -torch.inf)
            )
        if return_auxiliary:
            return logits, auxiliary_logits
        return logits


class ChemicalSpaceGroupEnsemble(nn.Module):
    """Average neural predictions without empirical priors or retrieval."""

    def __init__(self, checkpoints, weights=None):
        super().__init__()
        self.num_elements = checkpoints[0]["config"]["num_elements"]
        self.max_num_atoms = 54
        self.members = nn.ModuleList()
        for checkpoint in checkpoints:
            if checkpoint.get("prior_weight", 0) or checkpoint.get("prior") is not None:
                raise ValueError(
                    "Space-group inference requires purely neural checkpoints."
                )
            member = ChemicalSpaceGroupPredictor(**checkpoint["config"])
            member.load_state_dict(checkpoint["state_dict"], strict=True)
            self.members.append(member)
        weights = (
            torch.ones(len(checkpoints))
            if weights is None
            else torch.tensor(weights, dtype=torch.float32)
        )
        if (
            weights.shape != (len(checkpoints),)
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            raise ValueError(
                "Ensemble weights must be finite, nonnegative and have positive total."
            )
        self.register_buffer("weights", weights / weights.sum())

    @torch.inference_mode()
    def predict_space_groups(self, composition):
        composition = composition.to(next(self.parameters()).device)
        feasible = formula_space_group_mask(composition, self.max_num_atoms)
        probabilities = []
        for member in self.members:
            logits = member(composition, feasible)
            probabilities.append(logits.softmax(-1))
        return (torch.stack(probabilities) * self.weights[:, None, None]).sum(0)


def load_chemical_space_group_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("prior_exponent", 0) or checkpoint.get("prior") is not None:
        raise ValueError("Space-group inference requires purely neural checkpoints.")
    if checkpoint.get("architecture") == "neural_flow_sg":
        from models.pl_models.neural_flow_sg import NeuralFlowSpaceGroupPredictor

        model = NeuralFlowSpaceGroupPredictor(checkpoint)
    elif checkpoint.get("architecture") == "neural_fusion":
        from models.pl_models.neural_fusion import NeuralSpaceGroupFusion

        model = NeuralSpaceGroupFusion(checkpoint)
    elif "members" in checkpoint:
        model = ChemicalSpaceGroupEnsemble(
            checkpoint["members"], checkpoint.get("weights")
        )
    else:
        model = ChemicalSpaceGroupEnsemble([checkpoint])
    return model.to(device).eval()
