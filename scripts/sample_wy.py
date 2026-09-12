import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from models.common.checkpoint import load_model
from models.common.composition import formula_to_counts
from models.common.lookup_tables import chemical_symbols


class SamplingData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "target_index":
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def _format_formula(counts):
    counts = counts.detach().cpu().reshape(-1)
    return "".join(
        symbol if count == 1 else f"{symbol}{count}"
        for atomic_number, (symbol, count) in enumerate(
            zip(
                chemical_symbols,
                (int(round(float(value))) for value in counts.tolist()),
            )
        )
        if atomic_number > 0 and count > 0
    )


def _sample_or_skip(batch, model, count_conserving):
    try:
        return model.sample(
            batch,
            count_conserving=count_conserving,
        ).to_data_list()
    except ValueError as error:
        if str(error) != "Space group cannot realize the requested composition":
            raise
        if isinstance(batch, Batch) and batch.num_graphs > 1:
            generated_samples = []
            for target in batch.to_data_list():
                generated_samples.extend(
                    _sample_or_skip(target, model, count_conserving)
                )
            return generated_samples

        formula = _format_formula(batch.formula)
        space_group = int(batch.space_group.reshape(-1)[0])
        tqdm.write(f"Skipping formula={formula}, space_group={space_group}: {error}")
        return []


@torch.inference_mode()
def flow(loader, model, count_conserving=True):
    """Generate Wyckoff graphs for every formula in ``loader``."""
    generated_samples = []
    with tqdm(total=len(loader.dataset), desc="Sampling", unit="target") as progress:
        for batch in loader:
            generated_samples.extend(_sample_or_skip(batch, model, count_conserving))
            progress.update(batch.num_graphs)
    return generated_samples


class SampleDataset(Dataset):
    def __init__(
        self,
        formula,
        num_evals,
        space_group=None,
        num_elements=118,
        target_index=0,
    ):
        super().__init__()
        self.num_evals = int(num_evals)
        self.space_group = int(space_group)
        self.formula = formula_to_counts(formula, num_elements)
        self.target_index = target_index

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return SamplingData(
            space_group=torch.tensor(self.space_group, dtype=torch.long),
            num_evals=torch.tensor(self.num_evals, dtype=torch.long),
            formula=self.formula.unsqueeze(0).clone(),
            target_index=torch.tensor(self.target_index, dtype=torch.long),
        )


def load_formula_tabular_file(formula_file):
    with open(formula_file, "r") as f:
        line = f.readline().strip()
        if "formula" not in line:
            print("First line inferred NOT a HEADER, assume no header line")
            header = None
        else:
            header = 0
    formula_tabular = pd.read_csv(formula_file, header=header)
    if header is None:
        columns = list(formula_tabular.columns)
        print("Assume first column as formulas")
        columns[0] = "formula"
        formula_tabular.columns = columns

    formula_list = formula_tabular["formula"].tolist()
    # assume any other columns are conditions
    keys = [
        key for key in formula_tabular.columns if key not in {"formula", "num_evals"}
    ]
    conditions_list = list(formula_tabular[keys].T.to_dict().values())

    return formula_list, conditions_list


def main(args):
    print("Loading model...")
    model_path = Path(args.model_path)
    model, _, cfg = load_model(model_path, load_data=False)
    num_elements = int(cfg.model.model_config.num_elements)

    if args.formula_file is not None:
        print(f"Trying reading sampling formulas from '{args.formula_file}'...")
        formula_list, conditions_list = load_formula_tabular_file(args.formula_file)
    else:
        formula_list = args.formula
        conditions_list = [{}] * len(formula_list)

    if torch.cuda.is_available():
        model.to("cuda")

    test_set = ConcatDataset(
        SampleDataset(
            formula,
            args.num_evals,
            space_group=conditions.get("space_group", args.space_group),
            num_elements=num_elements,
            target_index=conditions.get("target_index", target_index),
        )
        for target_index, (formula, conditions) in enumerate(
            zip(formula_list, conditions_list)
        )
    )
    print(test_set[0])
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    start_time = time.time()
    generated_samples = flow(
        test_loader,
        model,
        count_conserving=args.count_conserving,
    )
    stop_time = time.time()
    print("Model time:", stop_time - start_time)

    torch.save(
        {
            "time": stop_time - start_time,
            "generated_samples": generated_samples,
            "args": vars(args),
        },
        str(args.save_path) + ".pt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample formula-conditioned Wyckoff graphs"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--formula", action="append")
    parser.add_argument("--formula_file")
    parser.add_argument(
        "--num_evals",
        type=int,
        default=1,
        help="number of samples to generate for every formula",
    )
    parser.add_argument("--space_group", type=int)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--count_conserving",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enforce the conditioned composition",
    )
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()
    if args.formula_file is None and not args.formula:
        parser.error("provide --formula or --formula_file")
    if args.formula_file is not None and args.formula:
        parser.error("--formula and --formula_file are mutually exclusive")
    main(args)
