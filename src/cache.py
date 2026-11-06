import hashlib
import os
import time
from multiprocessing import Pool

from PIL import Image

from . import render
from .config import CORPUS, DATA, FONT
from .sequence import STRIDE, output_frames, required_frames, visual_order

font = None


def settings_key():
    settings = f"{FONT}|{render.HEIGHT}|{render.FONT_SIZE}|{render.ABOVE}|{render.BELOW}|{render.PAD}"
    return hashlib.sha1(settings.encode()).hexdigest()[:10]


CACHE = DATA / "renders" / settings_key()


def init_worker():
    global font
    font = render.load_font()


def render_one(job):
    index, sentence = job
    try:
        image = render.render_line(sentence, font)
    except ValueError:
        return index, None, "clipped"
    if output_frames(image.width, STRIDE) < required_frames(visual_order(sentence)):
        return index, None, "too few frames"
    image.save(CACHE / f"{index:06d}.png", optimize=False)
    return index, image.width, None


def load_index():
    rows = []
    for line in (CACHE / "index.tsv").read_text(encoding="utf-8").splitlines():
        index, width, label = line.split("\t", 2)
        rows.append((int(index), int(width), label))
    return rows


def load_clean(index):
    with Image.open(CACHE / f"{index:06d}.png") as image:
        return image.copy()


def main():
    sentences = [line.split("\t", 1)[1] for line in CORPUS.read_text(encoding="utf-8").splitlines()]
    CACHE.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    kept, skipped = [], {}
    with Pool(os.cpu_count(), initializer=init_worker) as pool:
        for index, width, reason in pool.imap_unordered(render_one, enumerate(sentences), chunksize=64):
            if reason:
                skipped[reason] = skipped.get(reason, 0) + 1
            else:
                kept.append((index, width))

    kept.sort()
    with open(CACHE / "index.tsv", "w", encoding="utf-8") as f:
        for index, width in kept:
            f.write(f"{index}\t{width}\t{sentences[index]}\n")

    size = sum(p.stat().st_size for p in CACHE.glob("*.png"))
    print(f"{CACHE}: {len(kept)} lines, {size / 2**20:.0f} MB, {time.perf_counter() - start:.0f} s")
    for reason, n in skipped.items():
        print(f"skipped ({reason}): {n}")


if __name__ == "__main__":
    main()
