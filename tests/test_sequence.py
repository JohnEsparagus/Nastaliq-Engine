import unittest

from PIL import Image

from src.sequence import DIRECTION, from_reading_order, required_frames, to_reading_order, visual_order

ALEF, BEH = "ا", "ب"
YEAR_LINE = "سال 2020 میں"


class TestRequiredFrames(unittest.TestCase):
    def test_no_repeats(self):
        self.assertEqual(required_frames("ab"), 2)
        self.assertEqual(required_frames("کیا"), 3)

    def test_repeats_need_a_blank_between(self):
        self.assertEqual(required_frames("aa"), 3)
        self.assertEqual(required_frames("aab"), 4)
        self.assertEqual(required_frames("aaa"), 5)

    def test_empty(self):
        self.assertEqual(required_frames(""), 0)

    def test_batch_matches_single(self):
        labels = ["aa", "a", "", "ab", "bba", "", "aaa", "ککی"]
        self.assertEqual(list(required_frames(labels)), [required_frames(s) for s in labels])


class TestReadingOrder(unittest.TestCase):
    def test_direction_chosen(self):
        self.assertIn(DIRECTION, ("flip_image", "reverse_label"))

    def test_first_frame_sees_first_symbol(self):
        image = Image.new("L", (10, 4), 255)
        image.putpixel((9, 0), 0)
        out_image, out_label = to_reading_order(image, ALEF + BEH)
        starts_left = out_image.getpixel((0, 0)) == 0
        self.assertEqual(out_label, ALEF + BEH if starts_left else BEH + ALEF)

    def test_digits_keep_their_order_on_the_page(self):
        self.assertEqual(visual_order(YEAR_LINE), "ںیم 2020 لاس")

    def test_brackets_are_mirrored_like_the_renderer(self):
        self.assertEqual(visual_order("(2007)"), "(2007)")
        self.assertEqual(visual_order("(" + ALEF + BEH + ")"), "(" + BEH + ALEF + ")")

    def test_round_trip(self):
        image = Image.new("L", (4, 4), 255)
        for label in [YEAR_LINE, "(2007) " + ALEF + BEH, "3.5 " + BEH, ALEF + "\u064f" + BEH]:
            self.assertEqual(from_reading_order(to_reading_order(image, label)[1]), label)


if __name__ == "__main__":
    unittest.main()
