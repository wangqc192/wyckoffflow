"""Lightning module for composition-to-space-group prediction."""

import torch.nn.functional as F

from .base import OptimizedLightningModule
from .count_conserving import formula_space_group_mask
from .spg_predictor import SpaceGroupPredictor


class SpaceGroupModule(OptimizedLightningModule):
    def __init__(self, model_config, optimizer_config):
        super().__init__(model_config, optimizer_config, "space_group")
        config = self.model_config
        self.num_elements = config["num_elements"]
        self.max_num_atoms = config["max_num_atoms"]
        self.use_feasibility_mask = config.get(
            "compatibility",
            config.get("spg_compatibility", False),
        )
        self.predictor = SpaceGroupPredictor(config)

    def forward(self, batch):
        composition = batch.composition
        target = batch.space_group.long()
        logits = self.predictor(composition)
        if self.use_feasibility_mask:
            feasible = formula_space_group_mask(composition, self.max_num_atoms)
            logits = logits.masked_fill(~feasible, float("-inf"))
        loss = F.cross_entropy(logits, target)
        ranking_logits = logits.clone()
        ranking_logits[:, 0] = float("-inf")
        top_five = ranking_logits.topk(5, dim=1).indices
        return {
            "loss": loss,
            "spg_top1": (top_five[:, 0] == target).float().mean(),
            "spg_top5": (top_five == target[:, None]).any(dim=1).float().mean(),
        }

    def _shared_step(self, batch, prefix):
        metrics = self(batch)
        for name, value in metrics.items():
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=prefix == "train",
                on_epoch=True,
                prog_bar=name == "loss",
                batch_size=batch.num_graphs,
                sync_dist=True,
            )
        return metrics["loss"]

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
