import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from models.pl_data.datamodule import CrystDataModule


class IndexedDataset(Dataset):
    prototype_keys = tuple(str(i // 4) for i in range(32))

    def __len__(self):
        return len(self.prototype_keys)

    def __getitem__(self, index):
        # Dataset augmentation also consumes the global RNG.
        torch.rand(3)
        return index


@pytest.mark.parametrize("alpha", [0.0, 0.5])
def test_shuffle_seed_keeps_epoch_order_independent_of_model_rng(alpha):
    def orders(global_seed, shuffle_seed):
        torch.manual_seed(global_seed)
        module = CrystDataModule(
            datasets={},
            num_workers=OmegaConf.create({"train": 0}),
            batch_size=OmegaConf.create({"train": 8}),
            prototype_sampling_alpha=alpha,
            shuffle_seed=shuffle_seed,
        )
        module.train_dataset = IndexedDataset()
        loader = module.train_dataloader()
        return [torch.cat(list(loader)).tolist() for _ in range(2)]

    first = orders(42, 42)
    assert first == orders(123, 42)
    assert first[0] != first[1]
    assert first != orders(42, 123)
