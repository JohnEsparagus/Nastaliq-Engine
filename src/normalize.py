import re
import unicodedata

LOOKALIKES = {
    "ك": "ک",
    "ي": "ی",
    "ى": "ی",
    "ه": "ہ",
    "ۀ": "ۂ",
    "ة": "ۃ",
}

DIGITS = {chr(base + i): str(i) for base in (0x0660, 0x06F0) for i in range(10)}

PUNCTUATION = {
    "?": "؟",
    ";": "؛",
    "…": "...",
    "'": "’",
    "`": "’",
    "´": "’",
    '"': "”",
}

INVISIBLE = "\u200b\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\ufeff\u00adـ"
FORMAT = {ch: "" for ch in INVISIBLE}
FORMAT["\u200c"] = " "

TABLE = str.maketrans({**LOOKALIKES, **DIGITS, **PUNCTUATION, **FORMAT})


def unfold_presentation_form(ch):
    if "\ufb50" <= ch <= "\ufdff" or "\ufe70" <= ch <= "\ufefe":
        folded = unicodedata.normalize("NFKC", ch).strip()
        if len(folded) == 1:
            return folded
    return ch


def normalize(text):
    text = "".join(unfold_presentation_form(ch) for ch in text)
    text = unicodedata.normalize("NFC", text)
    text = text.translate(TABLE)
    text = text.replace("ی\u0654", "ئ")
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()
