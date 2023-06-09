from typing import Union

import torch
from torch.utils.data import Dataset


class AppendableDataset(Dataset):
    def __init__(self):
        self._data: list[tuple[torch.Tensor, str]] = list()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item) -> Union[tuple[torch.Tensor, str], list[tuple[torch.Tensor, str]]]:
        if isinstance(item, int):
            return self._data[item]
        elif isinstance(item, list):
            return [self._data[i] for i in item]
        else:
            raise IndexError(f'How do I use {item} as an index?')

    def append(self, image: torch.Tensor, label: str) -> 'AppendableDataset':
        self._data.append((image, label))
        return self


class BucketDataset(AppendableDataset):
    def __init__(self):
        super().__init__()
        self._buckets: dict[str, list[torch.Tensor]] = dict()

    def __len__(self):
        return sum([len(bucket) for bucket in self._buckets.values()])

    # TODO: Try changing index to just choose which bucket to pull from, then get a random item from that bucket
    def __getitem__(self, index) -> tuple[torch.Tensor, str]:
        bucket_index = index
        for label, bucket in sorted(self._buckets.items()):
            if bucket_index >= len(bucket):
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
