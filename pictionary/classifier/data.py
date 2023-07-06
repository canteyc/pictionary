import random
from collections import OrderedDict

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


class BucketDataset(Dataset):
    def __init__(self, labels: list[str] = None, bucket_sampler: str = 'random'):
        super().__init__()
        self._buckets: OrderedDict[str, list[torch.Tensor]] = OrderedDict()
        if labels:
            # initialize buckets for predetermined labels
            for key in labels:
                self._buckets[key] = list()
        if bucket_sampler == 'random':
            self._bucket_sampler = self._random_sampler
        elif bucket_sampler == 'sequential':
            self._bucket_sampler = self._sequential_sampler
        else:
            raise ValueError(f'Unknown option for bucket sampler: {bucket_sampler}. Options are (random, sequential)')

    def __len__(self):
        if self._bucket_sampler == self._random_sampler:
            return len(self._buckets)
        else:
            return self.size()

    def size(self, label: str = None) -> int:
        if label is None:
            return sum([len(bucket) for bucket in self._buckets.values()])
        else:
            return len(self._buckets[label])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        return self._bucket_sampler(index)

    def append(self, image: torch.Tensor, label: str) -> 'BucketDataset':
        if label in self._buckets:
            self._buckets[label].append(image)
        else:
            self._buckets[label] = [image]
        return self

    def _random_sampler(self, index: int) -> tuple[torch.Tensor, str]:
        k = list(self._buckets.keys())[index]
        return random.choice(self._buckets[k]), k

    def _sequential_sampler(self, index: int) -> tuple[torch.Tensor, str]:
        mutable_index = index
        for k, images in self._buckets.items():
            if mutable_index >= len(images):
                mutable_index -= len(images)
                continue
            return images[mutable_index], k
        raise IndexError(f'{index} out of range for dataset of size {self.size()}')
