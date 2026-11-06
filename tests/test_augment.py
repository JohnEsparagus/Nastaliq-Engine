import unittest

from src.augment import RANGES, augment, geometry, perspective_coefficients, rng_for, touches_border
from src.render import HEIGHT, load_font, render_line

LONG = "آر ایس ایس فیڈز انضمام اور عبدالرزاق سمیت چھ کھلاڑیوں کے ڈومیسٹک کرکٹ کھیلنے پر پابندی۔ " * 3


class TestAugment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        font = load_font()
        cls.clean = render_line("اس حوالے سے طبی ماہرین نے شدید نکتہ چینی کی تھی۔", font)
        cls.long = render_line(LONG, font)
        cls.tall = render_line("م\u064fستحکم\u0612 ن\u0650ظام\u0650 تعلیم", font)

    def test_same_seed_same_pixels(self):
        a = augment(self.clean, rng_for(0, 7))
        b = augment(self.clean, rng_for(0, 7))
        self.assertEqual(a.tobytes(), b.tobytes())

    def test_different_index_different_pixels(self):
        self.assertNotEqual(augment(self.clean, rng_for(0, 1)).tobytes(), augment(self.clean, rng_for(0, 2)).tobytes())

    def test_height_and_mode_kept(self):
        for i in range(20):
            out = augment(self.long, rng_for(0, i))
            self.assertEqual((out.height, out.mode), (HEIGHT, "L"))

    def test_never_inverted(self):
        for i in range(30):
            pixels = sorted(augment(self.clean, rng_for(1, i)).getdata())
            self.assertGreater(pixels[len(pixels) // 2] - pixels[len(pixels) // 100], 50)

    def test_identity_transform(self):
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        for got, want in zip(perspective_coefficients(square, square), [1, 0, 0, 0, 1, 0, 0, 0]):
            self.assertAlmostEqual(got, want)

    def test_no_geometry_keeps_letter_size(self):
        out = geometry(self.clean, 0, [(0, 0)] * 4)
        self.assertEqual(out.height, HEIGHT)
        self.assertLess(abs(out.width - self.clean.width), 0.1 * self.clean.width)

    def test_worst_case_geometry_cuts_nothing(self):
        for image in (self.clean, self.long, self.tall):
            j = RANGES["corner_jitter"] * image.height
            for angle in (-RANGES["rotate_deg"], RANGES["rotate_deg"]):
                for offsets in ([(-j, -j), (j, -j), (j, j), (-j, j)], [(j, j), (-j, j), (-j, -j), (j, -j)]):
                    self.assertFalse(touches_border(geometry(image, angle, offsets)))


if __name__ == "__main__":
    unittest.main()
