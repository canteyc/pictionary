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

        self.assertEqual(len(dataset), 9)


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

        weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
        loader = DataLoader(dataset, sampler=WeightedRandomSampler(weights, 3))

        for batch, labels in loader:
            self.assertEqual(len(batch), 1)
        self.assertEqual(len(loader), 3)

    def test_batch_weighted_sampler(self):
        dataset = AppendableDataset()

        dataset.append(torch.tensor((1)), 'a')
        dataset.append(torch.tensor((2)), 'a')
        dataset.append(torch.tensor((3)), 'a')

        dataset.append(torch.tensor((4)), 'b')
        dataset.append(torch.tensor((5)), 'b')
        dataset.append(torch.tensor((6)), 'b')

        dataset.append(torch.tensor((7)), 'c')
        dataset.append(torch.tensor((8)), 'c')
        dataset.append(torch.tensor((9)), 'c')

        weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
        sampler = WeightedRandomSampler(weights, 3)
        loader = DataLoader(dataset, sampler=BatchSampler(sampler, batch_size=3, drop_last=False))

        for batch in loader:
            # For some reason the batch size ends up being the minimum of num_samples from the weighted sampler and batch_size from BatchSampler
            self.assertEqual(len(batch), 3)
        self.assertEqual(len(loader), 1)
