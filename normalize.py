"""Unicode normalisation for Urdu OCR labels.

Every label goes through normalize(): the synthetic corpus (A1), UPTI ground
truth (A8) and hand transcriptions of real photos (A9). Synthetic images are
rendered from the normalised text, so whatever this maps, the image and the
label always agree.
"""
import re
import unicodedata

# Arabic code points that render the same as an Urdu letter in a Nastaliq font.
LOOKALIKES = {
    "ك": "ک",  # ك Arabic kaf      -> ک keheh
    "ي": "ی",  # ي Arabic yeh      -> ی Farsi yeh
    "ى": "ی",  # ى alef maksura    -> ی
    "ه": "ہ",  # ه Arabic heh      -> ہ heh goal
    "ۀ": "ۂ",  # ۀ heh + yeh above -> ۂ heh goal + hamza (keeps the mark)
    "ة": "ۃ",  # ة teh marbuta     -> ۃ teh marbuta goal
}

# One digit set. The news corpus is ~99.9% Western digits, so fold the rest into 0-9.
DIGITS = {chr(base + i): str(i) for base in (0x0660, 0x06F0) for i in range(10)}

# ASCII punctuation the Nastaliq font has no glyph for, or that has an Urdu twin.
PUNCTUATION = {
    "?": "؟",       # ؟
    ";": "؛",       # ؛
    "…": "...",     # … (no glyph in Noto Nastaliq)
    "'": "’",       # ' (no glyph) -> ’
    "`": "’",
    "´": "’",
    '"': "”",       # " (no glyph) -> ”
}

# ZWNJ marks a visible break in joining. A person transcribing a photo types a
# space there, so the label does too.
FORMAT = {"‌": " "}
# Invisible characters with nothing on the page for CTC to align to.
for ch in "​‍‎‏‪‫‬‭‮⁦⁧⁨⁩﻿­ـ":
    FORMAT[ch] = ""

TABLE = str.maketrans({**LOOKALIKES, **DIGITS, **PUNCTUATION, **FORMAT})


def unfold_presentation_form(ch):
    """ﭽ (U+FB7D) -> چ. Leaves multi-letter ligatures like ﷺ alone."""
    if "ﭐ" <= ch <= "﷿" or "ﹰ" <= ch <= "﻾":
        folded = unicodedata.normalize("NFKC", ch).strip()
        if len(folded) == 1:
            return folded
    return ch


def normalize(text):
    text = "".join(unfold_presentation_form(ch) for ch in text)
    # NFC before mapping: ي + ٔ composes to ئ only while it is still Arabic yeh.
    text = unicodedata.normalize("NFC", text)
    text = text.translate(TABLE)
    # ی + ٔ has no canonical composition, but it is the same glyph as ئ.
    text = text.replace("یٔ", "ئ")
    # NFC again: ہ + ٔ -> ۂ can only compose after ه -> ہ.
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()
