import unittest

from unittest.mock import patch
from src.dataset import LineDataset

class TestDataset(unittest.TestCase):
    @patch("src.dataset.load_index", return_value=[(0, 0, "a")])
    @patch("src.dataset.load_chars", return_value=["a"])
    def test_len(self, *mocks):
        self.assertEqual(len(LineDataset()), 1)
        
    

if __name__ == "__main__":
    unittest.main()
