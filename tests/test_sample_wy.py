from torch_geometric.data import Batch

from scripts.sample_wy import SampleDataset


def test_sample_records_batch_target_index_as_plain_scalar():
    records = [
        SampleDataset("Ga4Te4", 20, space_group=194, target_index=0)[0],
        SampleDataset("Ni2Cu2", 20, space_group=65, target_index=1)[0],
    ]

    batch = Batch.from_data_list(records)

    assert batch.target_index.tolist() == [0, 1]
    assert batch.num_evals.tolist() == [20, 20]
    assert batch.space_group.tolist() == [194, 65]
