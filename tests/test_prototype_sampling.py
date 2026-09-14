from types import SimpleNamespace

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from models.pl_data.datamodule import CrystDataModule


class PrototypeDataset(Dataset):
    def __init__(self, prototype_keys):
        self.prototype_keys = tuple(prototype_keys)

    def __len__(self):
        return len(self.prototype_keys)

    def __getitem__(self, index):
        return torch.tensor(index)


def make_datamodule(alpha):
    datamodule = CrystDataModule(
        datasets={},
        num_workers=SimpleNamespace(train=0, val=0, test=0),
        batch_size=SimpleNamespace(train=2, val=2, test=2),
        prototype_sampling_alpha=alpha,
    )
    datamodule.train_dataset = PrototypeDataset(("common", "common", "common", "rare"))
    return datamodule


def test_prototype_sampling_uses_frequency_weights_and_fixed_draw_count():
    loader = make_datamodule(0.5).train_dataloader()

    assert isinstance(loader.sampler, WeightedRandomSampler)
    assert len(loader.sampler) == 4
    assert torch.allclose(
        loader.sampler.weights,
        torch.tensor(
            [1 / (3**0.5), 1 / (3**0.5), 1 / (3**0.5), 1.0],
            dtype=torch.double,
        ),
    )


def test_zero_alpha_keeps_uniform_shuffle_sampler():
    loader = make_datamodule(0).train_dataloader()

    assert not isinstance(loader.sampler, WeightedRandomSampler)
