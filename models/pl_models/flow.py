"""Categorical flow matching for Wyckoff-position generation."""

import hydra
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
from .base import OptimizedLightningModule, resolve_config
from .model_utils import (
    create_wyckoff_graph,
    create_x_matrix,
    get_degrees_of_freedom,
)


def categorical_flow_step(current, target_logits, jump_probability, *, greedy=False):
    if current.numel() == 0:
        return current
    if greedy:
        return target_logits.argmax(dim=-1)
    target = Categorical(logits=target_logits, validate_args=False).sample()
    jump = torch.rand(current.shape, device=current.device) < jump_probability
    return torch.where(jump, target, current)


class DiscreteFlowModule(OptimizedLightningModule):
    def __init__(
        self,
        optimizer_config,
        decoder,
        num_elements,
        max_num_atoms,
        flow_source,
        zero_df_loss_weight=1.0,
        inf_df_loss_weight=1.0,
        mask_loss_by_composition=True,
        conditional_composition=True,
        validation_seed=42,
        label_smoothing=0.0,
    ):
        super().__init__(optimizer_config, "discrete_flow")
        decoder = resolve_config(decoder)
        self.save_hyperparameters(ignore=["optimizer_config"])
        self.validation_seed = validation_seed
        self._validation_rng = None
        self.num_elements = num_elements
        self.max_num_atoms = max_num_atoms
        self.zero_df_loss_weight = float(zero_df_loss_weight)
        self.inf_df_loss_weight = float(inf_df_loss_weight)
        self.mask_loss_by_composition = mask_loss_by_composition
        self.label_smoothing = float(label_smoothing)
        if self.zero_df_loss_weight < 0 or self.inf_df_loss_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if not 0 <= self.label_smoothing <= 1:
            raise ValueError("label_smoothing must be between 0 and 1")
        if not conditional_composition:
            raise ValueError("DiscreteFlowModule requires conditional_composition=True")

        self.source_zero = CategoricalSource(
            flow_source,
            self.num_elements + 1,
            MP20_ZERO_DOF_DISTRIBUTION,
        )
        self.source_inf = CategoricalSource(
            flow_source,
            self.max_num_atoms + 1,
            MP20_INF_DOF_DISTRIBUTION,
        )
        self.decoder_config = decoder
        self.decoder = hydra.utils.instantiate(
            self.decoder_config,
            num_elements=self.num_elements,
            max_num_atoms=self.max_num_atoms,
            conditional_composition=conditional_composition,
            continuous_time=True,
            _recursive_=False,
        )
        if not mask_loss_by_composition and not getattr(
            self.decoder, "predict_all_elements", True
        ):
            raise ValueError("Unmasked loss requires decoder.predict_all_elements=True")

    def forward(self, batch):
        return self.flow_loss(batch, self.encode_composition(batch.composition))

    def encode_composition(self, composition):
        """Joint models override this to share static features across tasks and steps."""
        return None

    def decode(self, data, time, composition_features):
        if composition_features is None:
            return self.decoder(data, time)
        return self.decoder(data, time, composition_features=composition_features)

    def flow_loss(self, batch, composition_features=None):
        """Corrupt occupations and train the decoder against the clean template."""
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

        zero_logits, inf_logits = self.decode(data_t, time, composition_features)
        raw_inf_logits = inf_logits
        if self.mask_loss_by_composition:
            zero_logits, inf_logits, allowed = self._mask_logits(
                zero_logits,
                inf_logits,
                data_t,
            )
        else:
            # Absent elements are supervised as count zero, not excluded.
            allowed = torch.ones(
                (batch_size, self.num_elements), dtype=torch.bool, device=self.device
            )
        variables_per_graph = batch.num_0_dof + batch.num_inf_dof * allowed.sum(dim=1)

        zero_allowed = torch.ones_like(zero_logits, dtype=torch.bool)
        if self.mask_loss_by_composition:
            zero_allowed = data_t.composition[data_t.batch[data_t.zero_dof]] > 0
            zero_allowed[:, 0] = True
        zero_loss = self._cross_entropy(
            zero_logits,
            batch.x_0_dof.long(),
            reduction="none",
            allowed=zero_allowed,
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
            raw_inf_logits.flatten(0, 1),
            batch.x_inf_dof.flatten().long(),
            reduction="none",
            label_smoothing=self.label_smoothing,
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
            "loss": (
                self.zero_df_loss_weight * zero_loss
                + self.inf_df_loss_weight * inf_loss
            ),
            "zero_df_loss": zero_loss,
            "inf_df_loss": inf_loss,
        }

    def _cross_entropy(self, logits, targets, *, reduction, allowed):
        if self.label_smoothing == 0 or allowed.all():
            return F.cross_entropy(
                logits,
                targets,
                reduction=reduction,
                label_smoothing=self.label_smoothing,
            )
        masked_logits = logits.masked_fill(
            ~allowed,
            torch.finfo(logits.dtype).min,
        )
        log_probs = masked_logits.log_softmax(dim=-1)
        nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        smooth = -(log_probs * allowed).sum(dim=-1) / allowed.sum(dim=-1)
        return (1 - self.label_smoothing) * nll + self.label_smoothing * smooth

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

    @torch.inference_mode()
    def sample_logits(self, batch, flow_steps, *, greedy=False):
        """Run one trajectory per condition, returning final CPU logits.

        Greedy trajectories take the decoder argmax at every step, without
        random jumps. Initial states still come from the configured source.
        """
        if flow_steps <= 0:
            raise ValueError("flow_steps must be positive")
        formulas = batch.formula
        if formulas.ndim == 1:
            formulas = formulas.unsqueeze(0)
        space_groups = batch.space_group.reshape(-1)

        data_t = self._build_source_from_compositions(
            formulas,
            space_groups,
            (
                batch.target_index.reshape(-1)
                if hasattr(batch, "target_index")
                else None
            ),
            (
                batch.sampling_group.reshape(-1)
                if hasattr(batch, "sampling_group")
                else None
            ),
        )

        zero_indices = data_t.zero_dof.nonzero().flatten()
        inf_indices = (~data_t.zero_dof).nonzero().flatten()
        allowed = data_t.composition > 0
        allowed[:, 0] = True
        zero_mask = ~allowed[data_t.batch[zero_indices]]
        inf_mask = (~allowed[:, 1:][data_t.batch[inf_indices]]).unsqueeze(-1) & (
            torch.arange(self.max_num_atoms + 1, device=self.device) > 0
        )
        composition_features = self.encode_composition(data_t.composition)
        for step in range(flow_steps):
            time = torch.full(
                (data_t.num_graphs,),
                step / flow_steps,
                device=self.device,
            )
            zero_logits, inf_logits = self.decode(data_t, time, composition_features)
            zero_logits = zero_logits.masked_fill(zero_mask, float("-inf"))
            inf_logits = inf_logits.masked_fill(inf_mask, float("-inf"))
            if step == flow_steps - 1:
                break

            jump_probability = 1 / (flow_steps - step)
            data_t.x_0_dof = categorical_flow_step(
                data_t.x_0_dof,
                zero_logits,
                jump_probability,
                greedy=greedy,
            )
            data_t.x_inf_dof = categorical_flow_step(
                data_t.x_inf_dof,
                inf_logits,
                jump_probability,
                greedy=greedy,
            )
            data_t.x[zero_indices, 0] = data_t.x_0_dof.float()
            data_t.x[inf_indices, 1:] = data_t.x_inf_dof.float()
        return data_t.cpu(), zero_logits.cpu(), inf_logits.cpu()

    def _build_source_from_compositions(
        self,
        compositions,
        fixed_space_group,
        target_indices=None,
        sampling_groups=None,
    ):
        # Build graph metadata on CPU, then transfer and draw all source states
        # in one batch instead of synchronizing CUDA once per graph/attribute.
        compositions = compositions.cpu()
        if isinstance(fixed_space_group, torch.Tensor):
            space_groups = fixed_space_group.cpu().reshape(-1)
            if space_groups.numel() != compositions.shape[0]:
                raise ValueError("space_group must contain one value per composition")
        else:
            space_groups = torch.full(
                (compositions.shape[0],),
                int(fixed_space_group),
                dtype=torch.long,
            )
        graphs = []
        if target_indices is None:
            target_indices = [None] * compositions.shape[0]
        else:
            target_indices = torch.as_tensor(target_indices).cpu()
        if sampling_groups is None:
            sampling_groups = [None] * compositions.shape[0]
        else:
            sampling_groups = torch.as_tensor(sampling_groups).cpu()
        for composition, space_group, target_index, sampling_group in zip(
            compositions, space_groups, target_indices, sampling_groups
        ):
            space_group = int(space_group)
            degrees = get_degrees_of_freedom(space_group)
            num_zero = int((degrees == 0).sum())
            num_inf = int((degrees != 0).sum())
            graph = create_wyckoff_graph(
                space_group,
                torch.zeros(num_zero, dtype=torch.long),
                torch.zeros(num_inf, self.num_elements, dtype=torch.long),
            )
            graph.composition = composition.unsqueeze(0)
            if target_index is not None:
                graph.target_index = torch.as_tensor(
                    target_index,
                    dtype=torch.long,
                )
            if sampling_group is not None:
                graph.sampling_group = torch.as_tensor(
                    sampling_group,
                    dtype=torch.long,
                )
            graphs.append(graph)
        data = Batch.from_data_list(graphs).to(self.device)
        data.x_0_dof = self.source_zero.sample(data.x_0_dof.shape)
        data.x_inf_dof = self.source_inf.sample(data.x_inf_dof.shape)
        data.x = create_x_matrix(data.x_inf_dof, data.x_0_dof, data.zero_dof)
        return data

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
