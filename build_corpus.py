"""A1: build the normalised corpus and freeze the character set.

    python3 build_corpus.py

Writes data/corpus.tsv (source<TAB>sentence) and charset.json.
"""
import json
import re
from collections import Counter
from pathlib import Path

from PIL import ImageFont

from normalize import normalize

DATA = Path("data")
FONT = "/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Regular.ttf"
# A character needs this many occurrences to earn a CTC label. Sentences with
# rarer characters are dropped instead.
MIN_COUNT = 100

# Letters Sindhi uses and Urdu doesn't. The news sets have Sindhi mixed in.
SINDHI_LETTERS = re.compile(
    "[ٺٻٽٿڀڃڄڇڊڌ"
    "ڍڏڙڪڳڻ۽۾]"
)
# Sindhi "is / are", for the few Sindhi sentences with no Sindhi-only letter.
SINDHI_WORDS = re.compile(r"(^|\s)(آهي|آهن)(\s|[.۔]|$)")
LATIN = re.compile("[A-Za-z]")


def repair_mojibake(text):
    """The 2007 set is UTF-8 that was decoded as cp1252 and saved again.
    cp1252 leaves 5 bytes undefined; those came through as latin-1."""
    raw = bytearray()
    for ch in text:
        try:
            raw += ch.encode("cp1252")
        except UnicodeEncodeError:
            raw += ch.encode("latin-1")
    return raw.decode("utf-8")


def load_sentences():
    """Yields (source, raw sentence) from every Leipzig *-sentences.txt."""
    for path in sorted(DATA.glob("*/*-sentences.txt")):
        source = path.parent.name
        text = path.read_text(encoding="utf-8")
        if "2007" in source:
            text = repair_mojibake(text)
        for line in text.splitlines():
            _, sentence = line.split("\t", 1)
            yield source, sentence


def font_has_glyph():
    """Returns a check for whether the font draws a character, or falls back to .notdef."""
    font = ImageFont.truetype(FONT, 40)
    notdef = bytes(font.getmask(""))
    return lambda ch: ch == " " or bytes(font.getmask(ch)) != notdef


def main():
    stats = Counter()
    seen = set()
    rows = []
    for source, raw in load_sentences():
        stats["raw"] += 1
        if SINDHI_LETTERS.search(raw) or SINDHI_WORDS.search(raw):
            stats["dropped: sindhi"] += 1
            continue
        if LATIN.search(raw):
            stats["dropped: latin script"] += 1
            continue
        sentence = normalize(raw)
        if not sentence or sentence in seen:
            stats["dropped: duplicate"] += 1
            continue
        seen.add(sentence)
        rows.append((source, sentence))

    counts = Counter(ch for _, s in rows for ch in s)
    has_glyph = font_has_glyph()
    no_glyph = {ch for ch in counts if not has_glyph(ch)}
    rare = {ch for ch, n in counts.items() if n < MIN_COUNT} - no_glyph

    kept = []
    for source, sentence in rows:
        chars = set(sentence)
        if chars & no_glyph:
            stats["dropped: no glyph in font"] += 1
        elif chars & rare:
            stats["dropped: rare character"] += 1
        else:
            kept.append((source, sentence))
    stats["kept"] = len(kept)

    counts = Counter(ch for _, s in kept for ch in s)
    chars = sorted(counts)
    charset = {
        "note": "CTC blank is index 0; chars[i] is label i + 1",
        "font": FONT,
        "min_count": MIN_COUNT,
        "chars": chars,
        "counts": {ch: counts[ch] for ch in chars},
    }
    Path("charset.json").write_text(json.dumps(charset, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(DATA / "corpus.tsv", "w", encoding="utf-8") as f:
        for source, sentence in kept:
            f.write(f"{source}\t{sentence}\n")

    for key, n in stats.items():
        print(f"{key:28} {n:>6}")
    print(f"{'no glyph in font':28} {' '.join(sorted(no_glyph))}")
    print(f"{'rare (< MIN_COUNT)':28} {' '.join(sorted(rare))}")
    print(f"\nfrozen character set: {len(chars)} (+ blank)")


if __name__ == "__main__":
    main()
