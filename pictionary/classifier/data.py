import random
from collections import OrderedDict
from typing import Union

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
    def __init__(self, labels: list[str] = None):
        super().__init__()
        self._buckets: OrderedDict[str, list[torch.Tensor]] = OrderedDict()
        if labels:
            # initialize buckets for predetermined labels
            for key in labels:
                self._buckets[key] = list()

    def __len__(self):
        return len(self._buckets)

    def size(self):
        return sum([len(bucket) for bucket in self._buckets.values()])

    def __getitem__(self, index) -> tuple[torch.Tensor, str]:
        key = list(self._buckets.keys())[index]
        image = random.choice(self._buckets[key])
        return image, key

    def append(self, image: torch.Tensor, label: str) -> 'BucketDataset':
        if label in self._buckets:
            self._buckets[label].append(image)
        else:
            self._buckets[label] = [image]
        return self
