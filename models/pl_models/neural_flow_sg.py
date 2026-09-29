"""Composition-only SG scoring with a pretrained occupation network."""

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from models.pl_models.chemical_sg import (
    ChemicalCompositionFeatures,
    ChemicalSpaceGroupEnsemble,
    ResidualBlock,
)
from models.pl_models.count_conserving import formula_space_group_mask
from models.pl_models.flow import DiscreteFlowModule
from models.pl_models.neural_fusion import NeuralSpaceGroupFusion
from models.sampling import SamplingData


class FlowSpaceGroupScore(nn.Module):
    def __init__(self, mean, scale, dropout=0.4, num_elements=100):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)
        self.features = ChemicalCompositionFeatures(
            num_elements=num_elements, count_histogram=True
        )
        self.context = nn.Sequential(
            nn.Linear(self.features.output_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            ResidualBlock(128, dropout),
        )
        self.group_embedding = nn.Embedding(231, 32)
        self.score = nn.Sequential(
            nn.Linear(mean.numel() + 128 + 32 + 1, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, features, groups, probability, chemistry, valid_groups=None):
        latent = (features.float() - self.mean) / self.scale
        logp = probability.gather(1, groups).clamp_min(1e-30).log()
        context = self.context(chemistry)
        inputs = torch.cat(
            (
                latent,
                context[:, None].expand(-1, groups.shape[1], -1),
                self.group_embedding(groups),
                logp[..., None] / 5,
            ),
            -1,
        )
        correction = 4 * self.score(inputs).squeeze(-1).tanh()
        if valid_groups is not None:
            correction = correction.masked_fill(~valid_groups, 0)
        logits = (
            probability.clamp_min(1e-30)
            .log()
            .scatter_add(1, groups, correction.float())
        )
        return logits, correction


@torch.inference_mode()
def flow_group_features(flow, composition, groups, valid_groups=None):
    """Pool neural representations at t=0 before any occupation is generated."""
    if flow.hparams.flow_source != "zeros":
        raise ValueError("Neural SG features require an all-zero flow source.")
    valid_groups = (
        torch.ones_like(groups, dtype=torch.bool, device="cpu")
        if valid_groups is None else valid_groups.cpu()
    )
    conditions = [
        SamplingData(
            formula=formula[None],
            space_group=group,
            target_index=torch.tensor(index),
            sampling_group=torch.tensor(index * groups.shape[1] + rank),
        )
        for index, (formula, candidates) in enumerate(
            zip(composition.cpu(), groups.cpu())
        )
        for rank, group in enumerate(candidates)
        if valid_groups[index, rank]
    ]
    captured = []
    hook = flow.decoder.output_norm.register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach())
    )
    try:
        graph, _, _ = flow.sample_logits(Batch.from_data_list(conditions), flow_steps=1)
    finally:
        hook.remove()
    hidden = captured[0]
    graph_ids = graph.batch.to(hidden.device)
    mean = scatter(hidden, graph_ids, dim=0, dim_size=graph.num_graphs, reduce="mean")
    maximum = scatter(hidden, graph_ids, dim=0, dim_size=graph.num_graphs, reduce="max")
    # The scorer was fitted on float16 feature storage, with float32 normalization.
    pooled = torch.cat((mean, maximum), -1).half()
    result = pooled.new_zeros((len(composition), groups.shape[1], pooled.shape[-1]))
    result[valid_groups.to(pooled.device)] = pooled
    return result


class NeuralFlowSpaceGroupPredictor(nn.Module):
    """Bundle all neural weights; inference never requires a reference structure."""

    def __init__(self, checkpoint):
        super().__init__()
        flow_config = checkpoint["flow_hyperparameters"]
        if flow_config["flow_source"] != "zeros":
            raise ValueError("Neural SG features require an all-zero flow source.")
        self.reference = NeuralSpaceGroupFusion(checkpoint["reference"])
        self.base = ChemicalSpaceGroupEnsemble(
            checkpoint["base_members"], checkpoint["base_weights"]
        )
        self.num_elements = self.base.num_elements
        self.max_num_atoms = self.base.max_num_atoms
        self.flow = DiscreteFlowModule(
            **{key: value for key, value in flow_config.items() if key != "task"}
        )
        self.flow.load_state_dict(checkpoint["flow_state_dict"], strict=True)
        state = checkpoint["score_state_dict"]
        self.score = FlowSpaceGroupScore(
            state["mean"],
            state["scale"],
            num_elements=self.num_elements,
            **checkpoint["score_config"],
        )
        self.score.load_state_dict(state, strict=True)
        self.proposal_groups = checkpoint["proposal_groups"]
        self.mixture_weight = checkpoint["mixture_weight"]

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # A Lightning submodule tracks device/dtype separately from its tensors.
        parameter = next(self.flow.parameters())
        self.flow.to(device=parameter.device, dtype=parameter.dtype)
        return self

    @torch.inference_mode()
    def predict_space_groups(self, composition):
        return self.predict_space_groups_with_base(composition)[0]

    @torch.inference_mode()
    def predict_space_groups_with_base(self, composition):
        composition = composition.to(next(self.parameters()).device)
        batches = [self._predict_batch(batch) for batch in composition.split(32)]
        return torch.cat([value[0] for value in batches]), torch.cat(
            [value[1] for value in batches]
        )

    def _predict_batch(self, composition):
        reference, proposals = self.reference.predict_space_groups_with_base(
            composition
        )
        probability = self.base.predict_space_groups(composition)
        groups = probability.topk(self.proposal_groups, -1).indices
        feasible = formula_space_group_mask(composition, self.max_num_atoms)
        valid_groups = feasible.gather(1, groups) & (probability.gather(1, groups) > 0)
        latent = flow_group_features(self.flow, composition, groups, valid_groups)
        logits, _ = self.score(
            latent, groups, probability, self.score.features(composition), valid_groups
        )
        learned = logits.masked_fill(~feasible, -torch.inf).softmax(-1)
        result = (1 - self.mixture_weight) * reference + self.mixture_weight * learned
        return result, proposals
