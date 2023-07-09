import datetime
import os
from typing import Callable, Optional

from kivy.graphics import Line
import torch
from torchvision.transforms.functional import gaussian_blur

from .classifier import digit_learner


MNIST_SIZE = 28
RADIUS = torch.sqrt(torch.tensor(2))


def line_to_image(line: Line) -> Optional[torch.Tensor]:
    if not line.points:
        return None
    points_tensor = torch.tensor(line.points)
    num_points = len(line.points) // 2
    points_tensor = points_tensor.resize(num_points, 2)

    # snap lower left corner of the shape to the origin
    lower_left = points_tensor.min(dim=0).values
    points_tensor -= lower_left

    # normalize shape to fit in 1x1 square
    upper_right = points_tensor.max(dim=0).values
    points_tensor /= upper_right
    # actually, make it a bit smaller than that and centered in 1x1 square so nothing is cut off in the low res image
    points_tensor *= 0.9
    points_tensor += 0.05

    image = torch.zeros(MNIST_SIZE, MNIST_SIZE)
    points_tensor *= MNIST_SIZE  # this is how big the actual image should be

    # Each point is surrounded by 4 pixel posts. This is the lower left one.
    lower_left_pixels = points_tensor.floor().int().unsqueeze(1)

    # Now we have all 4 pixels around each point, with shape (num_points, 4, 2)
    adjacent_pixels = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]]).repeat((num_points, 1, 1))
    pixels_around_points = lower_left_pixels + adjacent_pixels

    # flatten to (num_points * 4, 2), like a list of 2D points
    pixels = pixels_around_points.flatten(0, 1)
    points_repeated_four_times = points_tensor.unsqueeze(1).repeat(1, 4, 1).flatten(0, 1)
    distances = torch.linalg.norm(points_repeated_four_times - pixels, dim=1)  # shape(numpoints * 4)
    illuminations = (RADIUS - distances).clamp(0)  # clamp to remove negative light
    for pixel, light in zip(pixels, illuminations):
        image[pixel[1], pixel[0]] += light

    # sometimes a lot of points end up close to each other and really light up one pixel
    # I don't think that information is useful, so even out the image
    image[image > 0.1] = 1
    image = image.unsqueeze(0)
    image = gaussian_blur(image, [3, 3])  # fuzzier lines look more realistic in low resolution
    return image


class Agent:
    data_folder = f'../data'

    def __init__(self, set_text: Callable[[str], None]):
        self._brain = digit_learner()
        self.answer = None
        self.set_text = set_text
        self._load_data_from_disk()

    def store_image(self, drawing: Line, label: str):
        image = line_to_image(drawing)
        if image is None:
            return
        self.answer = self._send_to_model(image, label)
        self.set_text(f'This looks like a {self.answer}')
        self._save_to_disk(image, label)

    def accuracy(self, obj):
        accuracy_as_str_list = ''.join(f'{str(number):>7s}' for number in self._brain.accuracy().tolist())
        labels = ''.join(f'{label:>7s}' for label in self._brain.class_labels)
        self.set_text(f'{labels}\n{accuracy_as_str_list}')

    def _send_to_model(self, image: torch.Tensor, label: str) -> str:
        self._brain.add_image(image, label)
        self._brain.train()
        self.answer = self._brain.classify(image)
        return self.answer

    @staticmethod
    def _save_to_disk(image: torch.Tensor, label: str):
        date = datetime.datetime.now()
        timestamp = str(date).replace(' ', '-').replace(':', '-').replace('.', '-')
        filepath = f'{Agent.data_folder}/{timestamp}_{label}.pt'
        torch.save(image, filepath)

    def _load_data_from_disk(self):
        all_data = list(os.listdir(Agent.data_folder))
        for file in sorted(all_data):
            image = torch.load(f'{Agent.data_folder}/{file}')
            label = file.split('_')[-1].split('.')[0]
            self._brain.add_image(image, label)
