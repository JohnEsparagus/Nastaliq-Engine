"""A2: render a sentence as a single clean line image.

    python3 render.py      # writes out/samples.png and out/shaping.png

The image reads right to left, like the page. The label stays in logical
(typed) order; A5 decides whether to flip the image or reverse the labels.
"""
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from build_corpus import DATA, FONT

HEIGHT = 64      # final line height in px; A5 may raise it
FONT_SIZE = 48   # draw size, then scale down to HEIGHT
# Vertical band around the baseline, as a multiple of FONT_SIZE. The font
# claims 1.92 above / 0.60 below, but stacked ligatures with diacritics reach
# 2.73 above / 0.69 below (measured on 1500 corpus lines). Fixing the band,
# instead of cropping each line to its own ink, keeps letters the same size
# from line to line.
ABOVE = 2.9
BELOW = 0.8
PAD = 8          # px of white kept left and right of the ink, at FONT_SIZE


def load_font(shaped=True):
    """shaped=False uses Pillow's BASIC layout: no HarfBuzz, no joining, no bidi."""
    engine = ImageFont.Layout.RAQM if shaped else ImageFont.Layout.BASIC
    return ImageFont.truetype(FONT, FONT_SIZE, layout_engine=engine)


def render_line(text, font, height=HEIGHT):
    """Black on white, mode 'L', fixed height, width follows the text.
    Raises ValueError if the text draws nothing or its ink leaves the band."""
    above, below = round(ABOVE * FONT_SIZE), round(BELOW * FONT_SIZE)
    if font.layout_engine == ImageFont.Layout.RAQM:
        # language="ur" picks the Urdu forms of digits like ۴ ۶ ۷ over the Arabic/Persian ones.
        options = {"direction": "rtl", "language": "ur"}
    else:
        options = {}

    left, _, right, _ = font.getbbox(text, anchor="ls", **options)
    # Nastaliq swashes can overhang the advance box, so leave room either side.
    margin = FONT_SIZE
    width = right - left + 2 * margin
    image = Image.new("L", (width, above + below), 255)
    ImageDraw.Draw(image).text((margin - left, above), text, font=font, fill=0, anchor="ls", **options)

    ink = ImageOps.invert(image).getbbox()
    if ink is None:
        raise ValueError(f"nothing drawn for {text!r}")
    if ink[1] == 0 or ink[3] == image.height:
        raise ValueError(f"ink clipped by the band for {text!r}")
    image = image.crop((max(ink[0] - PAD, 0), 0, min(ink[2] + PAD, width), image.height))

    scaled_width = max(1, round(image.width * height / image.height))
    return image.resize((scaled_width, height), Image.LANCZOS)


def stack(images, gap=6):
    """Right-aligned column of line images, for eyeballing."""
    width = max(im.width for im in images)
    sheet = Image.new("L", (width, sum(im.height + gap for im in images)), 200)
    y = 0
    for im in images:
        sheet.paste(im, (width - im.width, y))
        y += im.height + gap
    return sheet


def main():
    out = Path("out")
    out.mkdir(exist_ok=True)
    sentences = [line.split("\t", 1)[1] for line in (DATA / "corpus.tsv").read_text(encoding="utf-8").splitlines()]
    shaped, unshaped = load_font(shaped=True), load_font(shaped=False)

    random.seed(0)
    short = [s for s in sentences if len(s) < 60]
    samples = random.sample(short, 8)
    stack([render_line(s, shaped) for s in samples]).save(out / "samples.png")

    # Break it: same text, with and without shaping.
    rows = []
    for s in ["پاکستان", "کیا حال ہے؟", samples[0]]:
        rows += [render_line(s, shaped), render_line(s, unshaped)]
    stack(rows, gap=12).save(out / "shaping.png")

    clipped = 0
    for s in random.sample(sentences, 2000):
        try:
            render_line(s, shaped)
        except ValueError:
            clipped += 1
    print(f"wrote {out}/samples.png and {out}/shaping.png")
    print(f"lines rejected for clipping: {clipped} / 2000")


if __name__ == "__main__":
    main()
