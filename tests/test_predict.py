import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.predict import load_chars, load_model, prepare, read, to_tensor
from src.render import load_font, render_line
from src.sequence import HEIGHT


class TestPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.chars = load_chars()
        cls.model = load_model(cls.chars)

    def test_any_photo_becomes_a_model_input(self):
        photo = Image.new("RGB", (1200, 180), (230, 220, 200))
        image = prepare(photo)
        self.assertEqual((image.mode, image.height), ("L", HEIGHT))
        self.assertEqual(tuple(to_tensor(image).shape), (1, 1, HEIGHT, image.width))

    def test_light_text_on_dark_is_inverted(self):
        self.assertGreater(prepare(Image.new("L", (300, 50), 20)).getpixel((0, 0)), 128)

    def test_tiny_image_still_gives_a_frame(self):
        self.assertIsInstance(read(Image.new("L", (2, 50), 255), self.model, self.chars), str)

    def test_reads_a_rendered_line(self):
        text = read(render_line("کیا حال ہے", load_font()), self.model, self.chars)
        self.assertTrue(set(text) <= set(self.chars))

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "crnn.pt"
            torch.save({"model": self.model.state_dict()}, path)
            loaded = load_model(self.chars, path)
        x = to_tensor(prepare(Image.new("L", (400, 64), 255)))
        with torch.no_grad():
            self.assertTrue(torch.equal(self.model(x), loaded(x)))


if __name__ == "__main__":
    unittest.main()
