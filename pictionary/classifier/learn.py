import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .model import LeNet


class ActiveLearner:
    def __init__(self, model: torch.nn.Module, class_labels: list[str]):
        self.model = model
        self.training_data: dict[str, DataLoader] = {class_: None for class_ in class_labels}
        self.test_data: dict[str, DataLoader] = {class_: None for class_ in class_labels}

    def add_image(self, image: torch.Tensor, label: str):
        if label not in self.training_data:
            raise KeyError(f"Label '{label}' not found in {self.training_data.keys()}")

        single_image_dataset = TensorDataset(image, torch.tensor([1]))
        if self.training_data[label] is None:
            self.training_data[label] = DataLoader(single_image_dataset)
        else:
            self.training_data[label].dataset += single_image_dataset

    @staticmethod
    def digit_learner():
        return ActiveLearner(LeNet(1, 10), [str(num) for num in range(10)])
