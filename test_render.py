import unittest

from render import HEIGHT, load_font, render_line


class TestRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shaped = load_font(shaped=True)
        cls.unshaped = load_font(shaped=False)

    def test_fixed_height_grayscale(self):
        image = render_line("پاکستان", self.shaped)
        self.assertEqual(image.height, HEIGHT)
        self.assertEqual(image.mode, "L")

    def test_width_grows_with_text(self):
        self.assertLess(render_line("کیا", self.shaped).width, render_line("کیا حال ہے؟", self.shaped).width)

    def test_letter_size_does_not_depend_on_the_line(self):
        # A tall word next to a flat one must not shrink the flat one.
        flat = render_line("دو", self.shaped)
        both = render_line("دو مُستحکم", self.shaped)
        self.assertEqual(flat.height, both.height)

    def test_shaping_changes_the_image(self):
        self.assertNotEqual(
            render_line("پاکستان", self.shaped).tobytes(),
            render_line("پاکستان", self.unshaped).tobytes(),
        )

    def test_lookalikes_draw_the_same(self):
        # The A1 trap: different code points, same pixels.
        self.assertEqual(render_line("كيا", self.shaped).tobytes(), render_line("کیا", self.shaped).tobytes())

    def test_blank_text_raises(self):
        with self.assertRaises(ValueError):
            render_line(" ", self.shaped)


if __name__ == "__main__":
    unittest.main()
