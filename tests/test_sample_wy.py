from torch.utils.data import ConcatDataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from scripts.sample_wy import SampleDataset, flow


def test_sample_records_batch_target_index_as_plain_scalar():
    records = [
        SampleDataset("Ga4Te4", 20, space_group=194, target_index=0)[0],
        SampleDataset("Ni2Cu2", 20, space_group=65, target_index=1)[0],
    ]

    batch = Batch.from_data_list(records)

    assert batch.target_index.tolist() == [0, 1]
    assert batch.num_evals.tolist() == [20, 20]
    assert batch.space_group.tolist() == [194, 65]


def test_flow_passes_inference_step_override():
    class FakeModel:
        def __init__(self):
            self.calls = []

        def sample(self, batch, count_conserving, flow_steps):
            self.calls.append((count_conserving, flow_steps))
            return batch

    dataset = ConcatDataset([SampleDataset("Ga4Te4", 2, space_group=194)])
    model = FakeModel()
    generated = flow(
        DataLoader(dataset, batch_size=1),
        model,
        count_conserving=True,
        flow_steps=50,
    )

    assert len(generated) == 1
    assert model.calls == [(True, 50)]
