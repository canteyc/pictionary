import unittest

import torch

from pictionary.classifier.learn import ActiveLearner


class TestActiveLearner(unittest.TestCase):
    def setUp(self) -> None:
        self.learner = ActiveLearner.digit_learner()

    def test_add_wrong_label(self):
        self.assertRaises(KeyError, self.learner.add_image, torch.Tensor(), 'does not exist')

    def test_add_image(self):
        image = torch.zeros((1, 28, 28))
        label = '0'

        self.learner.add_image(image, label)

        self.assertEqual(len(self.learner.training_data[label].dataset), 1)
