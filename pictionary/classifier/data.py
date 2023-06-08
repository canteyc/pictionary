import torch
from torch.utils.data import Dataset


class BucketDataset(Dataset):
    def __init__(self):
        self._buckets: dict[str, list[torch.Tensor]] = dict()

    def __len__(self):
        return sum([len(bucket) for bucket in self._buckets.values()])

    # TODO: Try changing index to just choose which bucket to pull from, then get a random item from that bucket
    def __getitem__(self, index) -> tuple[torch.Tensor, str]:
        bucket_index = index
        for label, bucket in sorted(self._buckets.items()):
            if bucket_index > len(bucket):
                bucket_index -= len(bucket)
                continue

            return bucket[bucket_index], label
        raise IndexError(f'{index} not found in BucketDataset with len {len(self)}')

    def append(self, image: torch.Tensor, label: str) -> 'BucketDataset':
        if label in self._buckets:
            self._buckets[label].append(image)
        else:
            self._buckets[label] = [image]
        return self
