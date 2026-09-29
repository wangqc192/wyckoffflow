"""Periodic validation reconstruction with equivalent Wyckoff templates."""

import csv
import json
import logging
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning import Callback
from pytorch_lightning.trainer.states import TrainerFn
from torch_geometric.data import Batch

from models.common.wyckoff_template import WyckoffTemplate
from models.pl_models.count_conserving import cpu_dp_pool, decode_composition_logits
from models.sampling import (
    SamplingData,
    decode_independent_logits,
    predict_space_group_conditions,
)

log = logging.getLogger(__name__)


def _record_samples(samples, target_keys, details, writer, sample_index):
    for sample in samples:
        index = int(sample.target_index)
        row = details[index]
        if sample.x.any():
            template = WyckoffTemplate.from_model_output(sample)
            formula = template.formula
            occupancy = template.to_crystalflow()
            matched = template.occupancy_key() in target_keys[index]
        else:
            # Empty unconstrained samples are retained as misses, as in eval_gwa.
            formula, occupancy, matched = "", str(int(sample.space_group)), False
        composition_correct = formula == row["formula"]
        row["generated_count"] += 1
        row["composition_correct_count"] += int(composition_correct)
        row["matched"] |= matched
        if row["generated_count"] == 1:
            row["matched_top1"] = matched
        writer.writerow(
            {
                "sample_index": sample_index,
                "target_index": index,
                "candidate_rank": row["generated_count"],
                "space_group": int(sample.space_group),
                "formula": formula,
                "target_formula": row["formula"],
                "wyckoff_occupancy": occupancy,
                "composition_correct": composition_correct,
                "count": 1,
            }
        )
        sample_index += 1
    return sample_index


@torch.inference_mode()
def evaluate_reconstruction(
    model,
    dataset,
    output_dir,
    *,
    num_samples,
    flow_steps,
    batch_size,
    cpu_workers,
    predicted_space_groups=0,
):
    """Compare constrained and unconstrained decoding of the same trajectories.

    Use true space groups by default, or distribute the same total sample budget
    across the top predicted groups. Missing samples remain misses.
    """
    output_dir = Path(output_dir)
    started = time.monotonic()
    directories = {True: output_dir, False: output_dir / "no_composition"}
    details = {mode: [] for mode in directories}
    sample_indices = {mode: 0 for mode in directories}
    # batch_size limits trajectories, including all repeats for each target.
    targets_per_batch = max(1, batch_size // num_samples)
    with ExitStack() as stack:
        executor, workers = stack.enter_context(
            cpu_dp_pool(cpu_workers, len(dataset) * num_samples)
        )
        writers = {}
        for mode, directory in directories.items():
            directory.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(
                (directory / "samples.csv").open("w", newline="")
            )
            writers[mode] = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_index",
                    "target_index",
                    "candidate_rank",
                    "space_group",
                    "formula",
                    "target_formula",
                    "wyckoff_occupancy",
                    "composition_correct",
                    "count",
                ],
            )
            writers[mode].writeheader()
        for start in range(0, len(dataset), targets_per_batch):
            stop = min(start + targets_per_batch, len(dataset))
            conditions, target_keys = [], {}
            for index in range(start, stop):
                graph = dataset[index]
                templates = WyckoffTemplate.from_protostructure_set(graph.aflow_label)
                target_keys[index] = {
                    template.occupancy_key() for template in templates
                }
                for rows in details.values():
                    rows.append(
                        {
                            "target_index": index,
                            "formula": templates[0].formula,
                            "space_group": int(graph.space_group),
                            "generated_count": 0,
                            "composition_correct_count": 0,
                            "matched_top1": False,
                            "matched": False,
                        }
                    )
                condition = SamplingData(
                    formula=graph.composition,
                    space_group=graph.space_group,
                    target_index=torch.tensor(index),
                    sampling_group=torch.tensor(index),
                )
                conditions.append(condition)
            if predicted_space_groups:
                conditions = predict_space_group_conditions(
                    model,
                    torch.cat([condition.formula for condition in conditions]),
                    range(start, stop),
                    predicted_space_groups,
                )
            group_counts = Counter(
                int(condition.target_index) for condition in conditions
            )
            group_ranks = Counter()
            trajectories = []
            for condition in conditions:
                index = int(condition.target_index)
                repeats, extra = divmod(num_samples, group_counts[index])
                repeats += group_ranks[index] < extra
                group_ranks[index] += 1
                trajectories.extend(condition.clone() for _ in range(repeats))
            data, zero_logits, inf_logits = model.sample_logits(
                Batch.from_data_list(trajectories), flow_steps=flow_steps
            )
            decoded = decode_composition_logits(
                data,
                zero_logits,
                inf_logits,
                model.max_num_atoms,
                stochastic=True,
                fixed_site_beam_size=max(256, 8 * num_samples),
                cpu_workers=workers,
                executor=executor,
            )
            # Independent decoding consumes only CPU randomness. Keep it from
            # changing subsequent flow trajectories or constrained samples.
            with torch.random.fork_rng(devices=[]):
                independent = decode_independent_logits(data, zero_logits, inf_logits)
            for mode, result in ((True, decoded), (False, independent)):
                sample_indices[mode] = _record_samples(
                    result.samples,
                    target_keys,
                    details[mode],
                    writers[mode],
                    sample_indices[mode],
                )
            if start == 0 or stop == len(dataset) or start // 1000 != stop // 1000:
                log.info("Validation reconstruction: %d/%d targets", stop, len(dataset))

    summaries = {}
    for mode, directory in directories.items():
        frame = pd.DataFrame(details[mode])
        frame.to_csv(directory / "details.csv", index=False)
        hits = int(frame.matched.sum())
        hits_top1 = int(frame.matched_top1.sum())
        correct = int(frame.composition_correct_count.sum())
        requested = len(dataset) * num_samples
        summaries[mode] = {
            "metric": "G-W-A Top-K",
            "top_k": num_samples,
            "predicted_space_groups": predicted_space_groups,
            "matched_materials": hits,
            "matched_materials_top1": hits_top1,
            "total_materials": len(dataset),
            "match_rate": hits / len(dataset),
            "match_rate_top1": hits_top1 / len(dataset),
            "materials_with_generated_samples": int((frame.generated_count > 0).sum()),
            "generated_samples": int(frame.generated_count.sum()),
            "requested_samples": requested,
            "composition_correct_samples": correct,
            "composition_accuracy": correct / requested,
            "flow_steps": flow_steps,
            "sampling_mode": "n-shot",
            "enforce_composition": mode,
            "batch_size": batch_size,
            "cpu_workers": cpu_workers if mode else 1,
            "elapsed_seconds": time.monotonic() - started,
        }
    return {**summaries[True], "no_composition": summaries[False]}


class ValidationReconstruction(Callback):
    """Run reconstruction on rank zero, then share metrics for checkpointing."""

    def __init__(
        self,
        output_dir,
        every_n_epochs=100,
        num_samples=20,
        flow_steps=50,
        batch_size=128,
        cpu_workers=8,
        seed=42,
        predicted_space_groups=0,
    ):
        super().__init__()
        if min(every_n_epochs, num_samples, flow_steps, batch_size, cpu_workers) < 1:
            raise ValueError(
                "reconstruction intervals and sampling sizes must be positive"
            )
        if batch_size < num_samples:
            raise ValueError("reconstruction batch_size must be at least num_samples")
        if not 0 <= predicted_space_groups <= min(num_samples, 230):
            raise ValueError(
                "predicted_space_groups must be in 0..min(num_samples, 230)"
            )
        self.output_dir = Path(output_dir)
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.flow_steps = flow_steps
        self.batch_size = batch_size
        self.cpu_workers = cpu_workers
        self.seed = seed
        self.predicted_space_groups = predicted_space_groups

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.state.fn != TrainerFn.FITTING:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs:
            return
        summary = None
        if trainer.is_global_zero:
            dataset = trainer.datamodule.val_datasets[0]
            directory = self.output_dir / f"epoch_{trainer.current_epoch:04d}"
            devices = (
                [pl_module.device.index or 0] if pl_module.device.type == "cuda" else []
            )
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(self.seed)
                if devices:
                    torch.cuda.default_generators[devices[0]].manual_seed(self.seed)
                summary = evaluate_reconstruction(
                    pl_module,
                    dataset,
                    directory,
                    num_samples=self.num_samples,
                    flow_steps=self.flow_steps,
                    batch_size=self.batch_size,
                    cpu_workers=self.cpu_workers,
                )
                if self.predicted_space_groups:
                    summary["joint"] = evaluate_reconstruction(
                        pl_module,
                        dataset,
                        directory / "joint",
                        num_samples=self.num_samples,
                        flow_steps=self.flow_steps,
                        batch_size=self.batch_size,
                        cpu_workers=self.cpu_workers,
                        predicted_space_groups=self.predicted_space_groups,
                    )
            metadata = dict(
                epoch=trainer.current_epoch,
                completed_epochs=trainer.current_epoch + 1,
                global_step=trainer.global_step,
                seed=self.seed,
            )
            summary.update(metadata)
            summary["no_composition"].update(metadata)
            if self.predicted_space_groups:
                summary["joint"].update(metadata)
                summary["joint"]["no_composition"].update(metadata)
                (directory / "joint/summary.json").write_text(
                    json.dumps(summary["joint"], indent=2) + "\n", encoding="utf-8"
                )
                (directory / "joint/no_composition/summary.json").write_text(
                    json.dumps(summary["joint"]["no_composition"], indent=2) + "\n",
                    encoding="utf-8",
                )
                log.info(
                    "Epoch %d joint reconstruction: GWA@1=%.2f%%, GWA@%d=%.2f%%",
                    trainer.current_epoch + 1,
                    100 * summary["joint"]["match_rate_top1"],
                    self.num_samples,
                    100 * summary["joint"]["match_rate"],
                )
            (directory / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            (directory / "no_composition" / "summary.json").write_text(
                json.dumps(summary["no_composition"], indent=2) + "\n", encoding="utf-8"
            )
            for mode, result in ((True, summary), (False, summary["no_composition"])):
                log.info(
                    "Epoch %d validation reconstruction (enforce_composition=%s): "
                    "GWA@1=%.2f%%, GWA@%d=%.2f%%, composition accuracy=%.2f%%",
                    trainer.current_epoch + 1,
                    mode,
                    100 * result["match_rate_top1"],
                    self.num_samples,
                    100 * result["match_rate"],
                    100 * result["composition_accuracy"],
                )
        summary = trainer.strategy.broadcast(summary, src=0)
        if self.predicted_space_groups:
            for name, value in {
                "joint_gwa_top1": summary["joint"]["match_rate_top1"],
                f"joint_gwa_top{self.num_samples}": summary["joint"]["match_rate"],
            }.items():
                pl_module.log(
                    f"val/{name}", value, on_step=False, on_epoch=True, sync_dist=True
                )
        for suffix, result in (
            ("", summary),
            ("_no_composition", summary["no_composition"]),
        ):
            for name, value in {
                "gwa_top1": result["match_rate_top1"],
                f"gwa_top{self.num_samples}": result["match_rate"],
                "composition_accuracy": result["composition_accuracy"],
            }.items():
                pl_module.log(
                    f"val/{name}{suffix}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
