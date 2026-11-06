import unittest

import torch

from src.model import CRNN, num_classes
from src.sequence import HEIGHT, output_frames


class TestCRNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.classes = num_classes()
        cls.model = CRNN(cls.classes).eval()

    def run_model(self, *shape):
        with torch.no_grad():
            return self.model(torch.rand(*shape))

    def test_plan_shape(self):
        self.assertEqual(tuple(self.run_model(2, 1, HEIGHT, 400).shape), (100, 2, self.classes))

    def test_frames_match_sequence_design(self):
        for width in (37, 401, 856, 1703):
            self.assertEqual(self.run_model(1, 1, HEIGHT, width).shape[0], int(output_frames(width)))

    def test_log_probs_sum_to_one(self):
        probs = self.run_model(2, 1, HEIGHT, 120).exp().sum(-1)
        self.assertTrue(torch.allclose(probs, torch.ones_like(probs), atol=1e-5))

    def test_cnn_squashes_height_to_one(self):
        with torch.no_grad():
            self.assertEqual(self.model.cnn(torch.rand(1, 1, HEIGHT, 64)).shape[2], 1)


if __name__ == "__main__":
    unittest.main()
