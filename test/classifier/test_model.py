import unittest

import torch

from pictionary.classifier.model import LeNet


class TestLeNet(unittest.TestCase):
    def setUp(self) -> None:
        self.model = LeNet(1, 10)

    def test_input_and_output_size(self):
        image = torch.zeros((1, 1, 28, 28))

        prediction = self.model(image)

        self.assertEqual(prediction.shape, (1, 10))

    def test_memorize_single_image(self):
        # two vertical bars on the edges
        image = torch.zeros((1, 1, 28, 28))
        image[0, 0, 0] += 1
        image[0, 0, -1] += 1
        label = torch.tensor([1])  # label our image as a 1

        # train the model on just this image 10 times
        loss_function = torch.nn.NLLLoss()
        opt = torch.optim.Adam(self.model.parameters())

        for _ in range(10):
            prediction = self.model(image)
            loss = loss_function(prediction, label)

            opt.zero_grad()
            loss.backward()
            opt.step()

        # the model should know this image is a 1 by now
        self.assertEqual(self.model.classify(image), 1)

