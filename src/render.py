import random

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .config import CORPUS, FONT, OUT
from .sequence import HEIGHT

FONT_SIZE = 48
ABOVE = 2.9
BELOW = 0.8
PAD = 8


def load_font(shaped=True):
    engine = ImageFont.Layout.RAQM if shaped else ImageFont.Layout.BASIC
    return ImageFont.truetype(FONT, FONT_SIZE, layout_engine=engine)


def render_line(text, font, height=HEIGHT):
    above, below = round(ABOVE * FONT_SIZE), round(BELOW * FONT_SIZE)
    if font.layout_engine == ImageFont.Layout.RAQM:
        options = {"direction": "rtl", "language": "ur"}
    else:
        options = {}

    left, _, right, _ = font.getbbox(text, anchor="ls", **options)
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
    width = max(im.width for im in images)
    sheet = Image.new("L", (width, sum(im.height + gap for im in images)), 200)
    y = 0
    for im in images:
        sheet.paste(im, (width - im.width, y))
        y += im.height + gap
    return sheet


def main():
    out = OUT
    out.mkdir(exist_ok=True)
    sentences = [line.split("\t", 1)[1] for line in CORPUS.read_text(encoding="utf-8").splitlines()]
    shaped, unshaped = load_font(shaped=True), load_font(shaped=False)

    random.seed(0)
    short = [s for s in sentences if len(s) < 60]
    samples = random.sample(short, 8)
    stack([render_line(s, shaped) for s in samples]).save(out / "samples.png")

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
