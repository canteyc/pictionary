import unittest

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler

from pictionary.classifier.data import BucketDataset, AppendableDataset


class TestBucketDataset(unittest.TestCase):
    def test_len_with_single_bucket(self):
        dataset = BucketDataset()

        dataset.append(torch.Tensor(()), '')

        self.assertEqual(len(dataset), 1)

    def test_len_with_multiple_buckets(self):
        dataset = BucketDataset()

        dataset.append(torch.Tensor(()), 'a')
        dataset.append(torch.Tensor(()), 'a')
        dataset.append(torch.Tensor(()), 'a')

        dataset.append(torch.Tensor(()), 'b')
        dataset.append(torch.Tensor(()), 'b')
        dataset.append(torch.Tensor(()), 'b')

        dataset.append(torch.Tensor(()), 'c')
        dataset.append(torch.Tensor(()), 'c')
        dataset.append(torch.Tensor(()), 'c')

        self.assertEqual(len(dataset), 3)


class TestWeightedDataLoader(unittest.TestCase):
    def test_batch_size(self):
        dataset = BucketDataset()

        dataset.append(torch.tensor((1)), 'a')
        dataset.append(torch.tensor((2)), 'a')
        dataset.append(torch.tensor((3)), 'a')

        dataset.append(torch.tensor((4)), 'b')
        dataset.append(torch.tensor((5)), 'b')
        dataset.append(torch.tensor((6)), 'b')

        dataset.append(torch.tensor((7)), 'c')
        dataset.append(torch.tensor((8)), 'c')
        dataset.append(torch.tensor((9)), 'c')

        weights = [0.1, 0.3, 0.6]
        batch_size = 5
        loader = DataLoader(dataset, sampler=WeightedRandomSampler(weights, batch_size), batch_size=batch_size)

        for batch, labels in loader:
            self.assertEqual(len(batch), batch_size)
        self.assertEqual(len(loader), 1)

    def test_batch_size_with_appendable_dataset(self):
        dataset = AppendableDataset()

        dataset.append(torch.tensor([1]), 'a')
        dataset.append(torch.tensor([2]), 'a')
        dataset.append(torch.tensor([3]), 'a')

        dataset.append(torch.tensor([4]), 'b')
        dataset.append(torch.tensor([5]), 'b')
        dataset.append(torch.tensor([6]), 'b')

        dataset.append(torch.tensor([7]), 'c')
        dataset.append(torch.tensor([8]), 'c')
        dataset.append(torch.tensor([9]), 'c')

        weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
        batch_size = 5
        sampler = WeightedRandomSampler(weights, batch_size)
        loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        for batch, _ in loader:
            self.assertEqual(batch.shape[0], batch_size)
        self.assertEqual(len(loader), 1)
