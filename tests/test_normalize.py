import unittest

from src.build_corpus import repair_mojibake
from src.normalize import normalize


class TestNormalize(unittest.TestCase):
    def test_arabic_lookalikes_become_urdu(self):
        self.assertEqual(normalize("آپ كے پاس كيا ہے؟"), "آپ کے پاس کیا ہے؟")
        self.assertEqual(normalize("الله"), "اللہ")

    def test_yeh_hamza_composes_before_and_after_mapping(self):
        self.assertEqual(normalize("ي\u0654"), "ئ")
        self.assertEqual(normalize("ی\u0654"), "ئ")

    def test_heh_hamza_composes_after_mapping(self):
        self.assertEqual(normalize("ه\u0654"), "ۂ")

    def test_alef_madda_composes(self):
        self.assertEqual(normalize("ا\u0653"), "آ")

    def test_diacritics_are_kept(self):
        self.assertEqual(normalize("م\u064fلک"), "م\u064fلک")

    def test_digits_fold_to_western(self):
        self.assertEqual(normalize("۲۰۲۰ ٢٠٢٠"), "2020 2020")

    def test_zwnj_is_a_space_and_invisibles_go(self):
        self.assertEqual(normalize("خوش\u200cآمدید"), "خوش آمدید")
        self.assertEqual(normalize("\u200fپاکستان\u200e"), "پاکستان")
        self.assertEqual(normalize("پاکـــستان"), "پاکستان")

    def test_presentation_forms_unfold(self):
        self.assertEqual(normalize("\ufb7d"), "چ")
        self.assertEqual(normalize("\ufe85"), "ؤ")

    def test_whitespace_collapses(self):
        self.assertEqual(normalize("  ایک   دو\t"), "ایک دو")

    def test_idempotent(self):
        for s in ["آپ كے پاس كيا ہے؟", "ي\u0654 ه\u0654", "خوش\u200cآمدید ۲۰۲۰…"]:
            self.assertEqual(normalize(normalize(s)), normalize(s))

    def test_repair_mojibake(self):
        def as_cp1252(b):
            try:
                return bytes([b]).decode("cp1252")
            except UnicodeDecodeError:
                return chr(b)

        broken = "".join(as_cp1252(b) for b in "ثانیہ".encode("utf-8"))
        self.assertEqual(repair_mojibake(broken), "ثانیہ")


if __name__ == "__main__":
    unittest.main()
