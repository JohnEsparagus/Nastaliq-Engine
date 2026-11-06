import sys

from PIL import Image

from src.augment import RANGES, augment, geometry, rng_for, touches_border
from src.config import CORPUS, OUT
from src.render import load_font, render_line, stack

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def main():
    sentences = [line.split("\t", 1)[1] for line in CORPUS.read_text(encoding="utf-8").splitlines()]
    font = load_font()
    rng = rng_for(SEED, "pick")
    OUT.mkdir(exist_ok=True)

    diacritics = set("\u064b\u064e\u064f\u0650\u0651\u0670")
    picks = rng.sample([s for s in sentences if len(s) < 70], 14)
    picks.append(max(sentences, key=len))
    picks.append(max(sentences, key=lambda s: sum(ch in diacritics for ch in s)))

    augmented = [augment(render_line(s, font), rng_for(SEED, i)) for i, s in enumerate(picks)]
    stack(augmented).save(OUT / "augment_sheet.png")
    with open(OUT / "augment_labels.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(picks):
            f.write(f"{i}\t{s}\n")

    pairs = []
    for i in range(4):
        clean = render_line(picks[i], font)
        for im in (clean, augmented[i]):
            pairs.append(im.resize((im.width * 2, im.height * 2), Image.NEAREST))
    stack(pairs).save(OUT / "augment_pairs.png")

    clean = render_line(picks[0], font)
    same = augment(clean, rng_for(SEED, 0)).tobytes() == augment(clean, rng_for(SEED, 0)).tobytes()

    longest = sorted(sentences, key=len)[-50:]
    stress = longest + rng.sample(sentences, 250)
    cut = 0
    for s in stress:
        clean = render_line(s, font)
        j = RANGES["corner_jitter"] * clean.height
        for angle in (-RANGES["rotate_deg"], RANGES["rotate_deg"]):
            out = geometry(clean, angle, [(-j, -j), (j, -j), (j, j), (-j, j)])
            cut += touches_border(out)

    print("wrote out/augment_sheet.png, out/augment_labels.txt, out/augment_pairs.png")
    print(f"reproducible with seed {SEED}: {same}")
    print(f"worst-case geometry, ink touching the edge: {cut} / {2 * len(stress)}")


if __name__ == "__main__":
    main()
