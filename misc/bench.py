import io
import os
import random
import sys
import time
from multiprocessing import Pool

from PIL import Image

from src.augment import augment, rng_for
from src.config import CORPUS
from src.render import load_font, render_line

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300

font = None


def init_worker():
    global font
    font = load_font()


def render_and_augment(job):
    index, sentence = job
    return augment(render_line(sentence, font), rng_for(0, index)).size


def per_line_ms(fn, items):
    start = time.perf_counter()
    for item in items:
        fn(item)
    return 1000 * (time.perf_counter() - start) / len(items)


def main():
    sentences = [line.split("\t", 1)[1] for line in CORPUS.read_text(encoding="utf-8").splitlines()]
    random.seed(0)
    sample = random.sample(sentences, N)
    init_worker()

    clean = [render_line(s, font) for s in sample]
    pngs = []
    for im in clean:
        buffer = io.BytesIO()
        im.save(buffer, "PNG")
        pngs.append(buffer.getvalue())

    render_ms = per_line_ms(lambda s: render_line(s, font), sample)
    augment_ms = per_line_ms(lambda i: augment(clean[i], rng_for(0, i)), range(N))
    decode_ms = per_line_ms(lambda b: Image.open(io.BytesIO(b)).load(), pngs)
    print(f"one core, per line:  render {render_ms:.1f} ms   augment {augment_ms:.1f} ms   "
          f"decode cached PNG {decode_ms:.2f} ms ({sum(map(len, pngs)) / N / 1024:.1f} KB each)")

    cores = os.cpu_count()
    jobs = list(enumerate(sample)) * 4
    with Pool(cores, initializer=init_worker) as pool:
        start = time.perf_counter()
        pool.map(render_and_augment, jobs, chunksize=16)
        rate = len(jobs) / (time.perf_counter() - start)
    print(f"{cores} cores, render + augment: {rate:.0f} lines/s")


if __name__ == "__main__":
    main()
