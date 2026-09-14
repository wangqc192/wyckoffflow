"""Categorical flow matching for Wyckoff-position generation."""

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..common.categorical import CategoricalSource
from ..common.dataset_info import (
    MP20_INF_DOF_DISTRIBUTION,
    MP20_ZERO_DOF_DISTRIBUTION,
)
from .base import OptimizedLightningModule
from .count_conserving import repair_batch_to_compositions
from .gnn import WyckoffGNN
from .model_utils import (
    create_wyckoff_graph,
    create_x_matrix,
    get_degrees_of_freedom,
)


def categorical_flow_step(current, target_logits, jump_probability):
    if current.numel() == 0:
        return current
    target = Categorical(logits=target_logits).sample()
    jump = torch.rand(current.shape, device=current.device) < jump_probability
    return torch.where(jump, target, current)


class DiscreteFlowModule(OptimizedLightningModule):
    def __init__(
        self,
        model_config,
        optimizer_config,
        validation_seed=42,
    ):
        super().__init__(model_config, optimizer_config, "discrete_flow")
        config = self.model_config
        self.validation_seed = validation_seed
        self.save_hyperparameters({"validation_seed": validation_seed})
        self._validation_rng = None
        self.num_elements = config["num_elements"]
        self.max_num_atoms = config["max_num_atoms"]
        self.flow_steps = config["flow_steps"]
        self.count_conserving = config.get("count_conserving", False)
        if not config["conditional_composition"]:
            raise ValueError("DiscreteFlowModule requires conditional_composition=True")

        self.source_zero = CategoricalSource(
            config["flow_source"],
            self.num_elements + 1,
            MP20_ZERO_DOF_DISTRIBUTION,
        )
        self.source_inf = CategoricalSource(
            config["flow_source"],
            self.max_num_atoms + 1,
            MP20_INF_DOF_DISTRIBUTION,
        )
        decoder_config = dict(config, continuous_time=True)
        self.decoder = WyckoffGNN(decoder_config)

    def forward(self, batch):
        data_t = batch.clone()
        batch_size = batch.num_graphs
        time = torch.rand(batch_size, device=self.device)

        time_zero = torch.repeat_interleave(time, batch.num_0_dof)
        data_t.x_0_dof = torch.where(
            torch.rand_like(time_zero) < time_zero,
            batch.x_0_dof,
            self.source_zero.sample(batch.x_0_dof.shape),
        )
        time_inf = torch.repeat_interleave(time, batch.num_inf_dof)
        data_t.x_inf_dof = torch.where(
            torch.rand(
                batch.x_inf_dof.shape,
                device=self.device,
            )
            < time_inf.unsqueeze(-1),
            batch.x_inf_dof,
            self.source_inf.sample(batch.x_inf_dof.shape),
        )
        data_t.x = create_x_matrix(
            data_t.x_inf_dof,
            data_t.x_0_dof,
            data_t.zero_dof,
        )

        zero_logits, inf_logits = self.decoder(data_t, time)
        zero_logits, inf_logits, allowed = self._mask_logits(
            zero_logits,
            inf_logits,
            data_t,
        )
        variables_per_graph = batch.num_0_dof + batch.num_inf_dof * allowed.sum(dim=1)

        zero_loss = F.cross_entropy(
            zero_logits,
            batch.x_0_dof.long(),
            reduction="none",
        )
        zero_loss = scatter(
            zero_loss,
            data_t.batch[data_t.zero_dof],
            dim=0,
            dim_size=batch_size,
            reduce="sum",
        )
        zero_loss = (zero_loss / variables_per_graph).mean()

        inf_loss = F.cross_entropy(
            inf_logits.flatten(0, 1),
            batch.x_inf_dof.flatten().long(),
            reduction="none",
        ).reshape(-1, self.num_elements)
        inf_graph = data_t.batch[~data_t.zero_dof]
        inf_loss = (inf_loss * allowed[inf_graph]).sum(dim=1)
        inf_loss = scatter(
            inf_loss,
            inf_graph,
            dim=0,
            dim_size=batch_size,
            reduce="sum",
        )
        inf_loss = (inf_loss / variables_per_graph).mean()

        return {
            "loss": zero_loss + inf_loss,
            "zero_df_loss": zero_loss,
            "inf_df_loss": inf_loss,
        }

    @staticmethod
    def _apply_element_mask(zero_logits, inf_logits, zero_allowed, inf_allowed):
        zero_logits = zero_logits.masked_fill(~zero_allowed, float("-inf"))
        positive_count = (
            torch.arange(inf_logits.shape[-1], device=inf_logits.device) > 0
        )
        inf_mask = (~inf_allowed).unsqueeze(-1) & positive_count
        return zero_logits, inf_logits.masked_fill(inf_mask, float("-inf"))

    def _mask_logits(self, zero_logits, inf_logits, data):
        allowed = data.composition.to(zero_logits.device) > 0
        allowed[:, 0] = True
        zero_graph = data.batch[data.zero_dof]
        inf_graph = data.batch[~data.zero_dof]
        zero_logits, inf_logits = self._apply_element_mask(
            zero_logits,
            inf_logits,
            allowed[zero_graph],
            allowed[:, 1:][inf_graph],
        )
        return zero_logits, inf_logits, allowed[:, 1:]

    _mask_training_logits = _mask_logits

    @torch.inference_mode()
    def sample(self, batch, count_conserving=None, flow_steps=None):
        """Sample graphs from formula-conditioned records in ``batch``."""
        required = ("formula", "num_evals", "space_group")
        missing = [name for name in required if not hasattr(batch, name)]
        if missing:
            raise ValueError(f"sampling batch is missing: {', '.join(missing)}")

        batch = batch.to(self.device)
        formulas = batch.formula
        if formulas.ndim == 1:
            formulas = formulas.unsqueeze(0)
        num_evals = int(batch.num_evals.reshape(-1)[0])
        if num_evals <= 0:
            raise ValueError("num_evals must be positive")
        space_groups = batch.space_group.reshape(-1)
        if space_groups.numel() != formulas.shape[0]:
            raise ValueError("space_group must contain one value per formula")
        if not torch.all((1 <= space_groups) & (space_groups <= 230)):
            raise ValueError("space_group must lie in 1..230")
        sample_flow_steps = self.flow_steps if flow_steps is None else int(flow_steps)
        if sample_flow_steps <= 0:
            raise ValueError("flow_steps must be positive")

        data_t = self._build_source_from_compositions(
            formulas.repeat_interleave(num_evals, dim=0),
            space_groups.repeat_interleave(num_evals),
            (
                batch.target_index.reshape(-1).repeat_interleave(num_evals)
                if hasattr(batch, "target_index")
                else None
            ),
        )

        for step in range(sample_flow_steps):
            time = torch.full(
                (data_t.num_graphs,),
                step / sample_flow_steps,
                device=self.device,
            )
            zero_logits, inf_logits = self.decoder(data_t, time)
            zero_logits, inf_logits, _ = self._mask_logits(
                zero_logits,
                inf_logits,
                data_t,
            )
            jump_probability = 1 / (sample_flow_steps - step)
            data_t.x_0_dof = categorical_flow_step(
                data_t.x_0_dof,
                zero_logits,
                jump_probability,
            )
            data_t.x_inf_dof = categorical_flow_step(
                data_t.x_inf_dof,
                inf_logits,
                jump_probability,
            )
            data_t.x = create_x_matrix(
                data_t.x_inf_dof,
                data_t.x_0_dof,
                data_t.zero_dof,
            )

        use_count_conserving = (
            self.count_conserving if count_conserving is None else count_conserving
        )
        if use_count_conserving:
            data_t, _ = repair_batch_to_compositions(
                data_t,
                zero_logits,
                inf_logits,
                self.max_num_atoms,
            )
            data_t.x = create_x_matrix(
                data_t.x_inf_dof,
                data_t.x_0_dof,
                data_t.zero_dof,
            )
        return data_t

    def _build_source_from_compositions(
        self,
        compositions,
        fixed_space_group,
        target_indices=None,
    ):
        compositions = compositions.to(self.device)
        if isinstance(fixed_space_group, torch.Tensor):
            space_groups = fixed_space_group.to(self.device).reshape(-1)
            if space_groups.numel() != compositions.shape[0]:
                raise ValueError("space_group must contain one value per composition")
        else:
            space_groups = torch.full(
                (compositions.shape[0],),
                int(fixed_space_group),
                device=self.device,
                dtype=torch.long,
            )
        graphs = []
        if target_indices is None:
            target_indices = [None] * compositions.shape[0]
        for composition, space_group, target_index in zip(
            compositions, space_groups, target_indices
        ):
            space_group = int(space_group)
            degrees = get_degrees_of_freedom(space_group, self.device)
            num_zero = int((degrees == 0).sum())
            num_inf = int((degrees != 0).sum())
            graph = create_wyckoff_graph(
                space_group,
                self.source_zero.sample((num_zero,)),
                self.source_inf.sample((num_inf, self.num_elements)),
            )
            graph.composition = composition.unsqueeze(0)
            if target_index is not None:
                graph.target_index = torch.as_tensor(
                    target_index,
                    device=self.device,
                    dtype=torch.long,
                )
            graphs.append(graph)
        return Batch.from_data_list(graphs)

    def _shared_step(self, batch, prefix):
        losses = self(batch)
        for name, value in losses.items():
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=prefix == "train",
                on_epoch=True,
                prog_bar=name == "loss",
                batch_size=batch.num_graphs,
                sync_dist=True,
            )
        return losses["loss"]

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def on_validation_epoch_start(self):
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        self._validation_rng = torch.random.fork_rng(devices=devices)
        self._validation_rng.__enter__()
        torch.manual_seed(self.validation_seed)

    def on_validation_epoch_end(self):
        self._validation_rng.__exit__(None, None, None)
        self._validation_rng = None
