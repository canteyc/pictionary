import random
from collections import OrderedDict
from typing import Union, Callable

import torch
from torch.utils.data import Dataset


class AppendableDataset(Dataset):
    def __init__(self):
        self._images: list[torch.Tensor] = list()
        self._labels: list[str] = list()

    def __len__(self) -> int:
        # Used by dataloader to determine max index for __getitem__
        return len(self._images)

    def size(self):
        # The number of images in the dataset
        return len(self)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, str]:
        return self._images[item], self._labels[item]

    def append(self, image: torch.Tensor, label: str) -> 'AppendableDataset':
        self._images.append(image)
        self._labels.append(label)
        return self


class BucketDataset(AppendableDataset):
    def __init__(self, labels: list[str] = None, bucket_sampler: str = 'random'):
        super().__init__()
        self._buckets: OrderedDict[str, list[torch.Tensor]] = OrderedDict()
        if labels:
            # initialize buckets for predetermined labels
            for key in labels:
                self._buckets[key] = list()
        if bucket_sampler == 'random':
            self._bucket_sampler = lambda k: random.choice(self._buckets[k])
        elif bucket_sampler == 'sequential':
            self._buckets_index = {k: 0 for k in self._buckets}
            self._bucket_sampler = lambda k: self._buckets[k][self._buckets_index[k]]
        else:
            raise ValueError(f'Unknown option for bucket sampler: {bucket_sampler}. Options are (random, sequential)')

    def __len__(self):
        return len(self._buckets)

    def size(self, label: str = None) -> int:
        if label is None:
            return sum([len(bucket) for bucket in self._buckets.values()])
        else:
            return len(self._buckets[label])

    def __getitem__(self, index) -> tuple[torch.Tensor, str]:
        key = list(self._buckets.keys())[index]
        image = self._bucket_sampler(key)
        return image, key

    def append(self, image: torch.Tensor, label: str) -> 'BucketDataset':
        if label in self._buckets:
            self._buckets[label].append(image)
        else:
            self._buckets[label] = [image]
            self._buckets_index[label] = 0
        return self
