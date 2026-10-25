import unittest

from build_corpus import repair_mojibake
from normalize import normalize


class TestNormalize(unittest.TestCase):
    def test_arabic_lookalikes_become_urdu(self):
        self.assertEqual(normalize("آپ كے پاس كيا ہے؟"), "آپ کے پاس کیا ہے؟")
        self.assertEqual(normalize("الله"), "اللہ")

    def test_yeh_hamza_composes_before_and_after_mapping(self):
        self.assertEqual(normalize("ئ"), "ئ")  # ي + ٔ (canonical)
        self.assertEqual(normalize("یٔ"), "ئ")  # ی + ٔ (no canonical form)

    def test_heh_hamza_composes_after_mapping(self):
        self.assertEqual(normalize("هٔ"), "ۂ")  # ه + ٔ -> ۂ

    def test_alef_madda_composes(self):
        self.assertEqual(normalize("آ"), "آ")  # ا + ٓ -> آ

    def test_diacritics_are_kept(self):
        self.assertEqual(normalize("مُلک"), "مُلک")

    def test_digits_fold_to_western(self):
        self.assertEqual(normalize("۲۰۲۰ ٢٠٢٠"), "2020 2020")

    def test_zwnj_is_a_space_and_invisibles_go(self):
        self.assertEqual(normalize("خوش‌آمدید"), "خوش آمدید")
        self.assertEqual(normalize("‏پاکستان‎"), "پاکستان")
        self.assertEqual(normalize("پاکـــستان"), "پاکستان")  # tatweel

    def test_presentation_forms_unfold(self):
        self.assertEqual(normalize("ﭽ"), "چ")  # ﭽ -> چ
        self.assertEqual(normalize("ﺅ"), "ؤ")  # ﺅ -> ؤ

    def test_whitespace_collapses(self):
        self.assertEqual(normalize("  ایک   دو\t"), "ایک دو")

    def test_idempotent(self):
        for s in ["آپ كے پاس كيا ہے؟", "ئ هٔ", "خوش‌آمدید ۲۰۲۰…"]:
            self.assertEqual(normalize(normalize(s)), normalize(s))

    def test_repair_mojibake(self):
        # Break it the way the 2007 set was broken. ہ is DB 81, and 0x81 is undefined in cp1252.
        def as_cp1252(b):
            try:
                return bytes([b]).decode("cp1252")
            except UnicodeDecodeError:
                return chr(b)

        broken = "".join(as_cp1252(b) for b in "ثانیہ".encode("utf-8"))
        self.assertEqual(repair_mojibake(broken), "ثانیہ")


if __name__ == "__main__":
    unittest.main()
