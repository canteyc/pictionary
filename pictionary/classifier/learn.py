from typing import List

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler

from .data import AppendableDataset
from .model import LeNet


class ActiveLearner:
    def __init__(self, model: torch.nn.Module, class_labels: list[str]):
        self.model = model
        self.class_labels = class_labels
        batch_size = 1
        weighted_sampler = WeightedRandomSampler([0.] * len(self.class_labels), batch_size)
        self.weights = weighted_sampler.weights
        self.training_data = DataLoader(AppendableDataset(), sampler=BatchSampler(weighted_sampler, batch_size, False))
        self.test_data = DataLoader(AppendableDataset())

        self._train_test_ratio = 4  # number of train samples vs test samples

    def add_image(self, image: torch.Tensor, label: str):
        if label not in self.class_labels:
            raise KeyError(f'{label} not in {self.class_labels}')

        image = self.reshape(image)
        if len(self.test_data.dataset) == 0 or len(self.training_data.dataset) / len(self.test_data.dataset) > self._train_test_ratio:
            self.test_data.dataset.append(image, label)
        else:
            self.training_data.dataset.append(image, label)

    def classify(self, image: torch.Tensor) -> str:
        return self.class_labels[self.model.classify(image.unsqueeze(0))]

    def accuracy(self, no_data_value=0) -> torch.Tensor:
        count = torch.zeros(len(self.class_labels))
        num_correct = torch.zeros_like(count)
        self.model.eval()
        with torch.no_grad():
            for batch, labels in self.test_data:
                indices = torch.tensor([self.class_labels.index(label) for label in labels])
                predictions = self.model.classify(batch)
                num_correct[indices] += predictions.item() == self.class_labels.index(labels[0])
                count[indices] += 1
        return torch.nan_to_num(num_correct / count, no_data_value)

    def _reset_training_sampler(self):
        accuracy = self.accuracy(no_data_value=1)
        self.weights[:] = 1. - accuracy

    def train(self):
        self._reset_training_sampler()

        self.model.train()


    @staticmethod
    def digit_learner():
        return ActiveLearner(LeNet(1, 10), [str(num) for num in range(10)])

    @staticmethod
    def reshape(image: torch.Tensor) -> torch.Tensor:
        if len(image.shape) < 3:
            image = image.unsqueeze(0)
        return image
