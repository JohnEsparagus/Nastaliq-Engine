import random
import sys

import numpy as np

from src.config import CORPUS
from src.render import HEIGHT as RENDER_HEIGHT, load_font, render_line
from src.sequence import output_frames, required_frames, visual_order

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
HEIGHTS = [48, 64, 80, 96]
STRIDES = [2, 4, 8, 16]


def main():
    sentences = [line.split("\t", 1)[1] for line in CORPUS.read_text(encoding="utf-8").splitlines()]
    random.seed(0)
    sample = random.sample(sentences, N)
    font = load_font()

    widths = np.array([render_line(s, font).width for s in sample])
    need = required_frames([visual_order(s) for s in sample])

    print(f"{N} lines. Cell = % that fit (T >= required), then worst T / required.\n")
    print("height  " + "".join(f"stride {s:<10}" for s in STRIDES))
    for h in HEIGHTS:
        scaled = np.rint(widths * h / RENDER_HEIGHT).astype(int)
        cells = []
        for stride in STRIDES:
            ratio = output_frames(scaled, stride) / need
            cells.append(f"{100 * np.mean(ratio >= 1):5.1f}% {ratio.min():4.2f}   ")
        print(f"{h:<8}" + "".join(cells))

    px_per_frame = widths / need
    print("\ntightest lines (px per required frame at render height):")
    for i in np.argsort(px_per_frame)[:5]:
        print(f"  {px_per_frame[i]:5.2f}  {sample[i]}")


if __name__ == "__main__":
    main()
