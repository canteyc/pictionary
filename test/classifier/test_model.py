import unittest

import torch

from pictionary.classifier.model import LeNet


class TestLeNet(unittest.TestCase):
    def test_input_and_output_size(self):
        image = torch.zeros((1, 28, 28))
        model = LeNet(1, 10)
        prediction = model(image)
        self.assertEqual(prediction.shape(), 10)
