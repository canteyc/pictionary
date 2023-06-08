from typing import List

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .model import LeNet


class ActiveLearner:
    def __init__(self, model: torch.nn.Module, class_labels: list[str]):
        self.model = model
        self.class_labels = class_labels
        self.training_data: dict[str, DataLoader] = {class_: None for class_ in class_labels}
        self.test_data: dict[str, DataLoader] = {class_: None for class_ in class_labels}

        self._train_test_ratio = 4  # number of train samples vs test samples

    def add_image(self, image: torch.Tensor, label: str):
        if label not in self.training_data:
            raise KeyError(f"Label '{label}' not found in {self.training_data.keys()}")

        image = self.reshape(image)
        single_image_dataset = TensorDataset(image, torch.tensor([label]))
        if self.test_data[label] is None:
            self.test_data[label] = DataLoader(single_image_dataset)
        elif self.training_data[label] is None:
            self.training_data[label] = DataLoader(single_image_dataset)
        elif len(self.training_data[label].dataset) / len(self.test_data[label].dataset) < self._train_test_ratio:
            self.training_data[label].dataset += single_image_dataset
        else:
            self.test_data[label].dataset += single_image_dataset

    def classify(self, image: torch.Tensor) -> str:
        return self.class_labels[self.model.classify(self.reshape(image))]

    def accuracy(self) -> List[float]:
        acc = [0.] * len(self.class_labels)
        for key in self.test_data:
            acc[int(key)] = self.class_accuracy(key)
        return acc

    def class_accuracy(self, label: str) -> float:
        self.model.eval()
        num_correct = 0
        loader = self.test_data[label]
        if loader is None:
            return 0.

        with torch.no_grad():
            for batch, answer in loader:
                predictions = self.model.classify(batch)
                num_correct += (predictions == answer).type(torch.float).sum().item()
        return num_correct / len(loader.dataset)


    @staticmethod
    def digit_learner():
        return ActiveLearner(LeNet(1, 10), [str(num) for num in range(10)])

    @staticmethod
    def reshape(image: torch.Tensor) -> torch.Tensor:
        while len(image.shape) < 4:
            image = image.unsqueeze(0)
        return image
