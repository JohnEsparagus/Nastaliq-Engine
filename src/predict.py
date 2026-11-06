import argparse
import json

import torch
from PIL import Image, ImageOps

from .augment import augment, rng_for
from .config import CHARSET, OUT
from .ctc import greedy_decode
from .model import CRNN
from .render import load_font, render_line
from .sequence import HEIGHT, STRIDE, from_reading_order, to_reading_order


def load_chars():
    with open(CHARSET, encoding="utf-8") as f:
        return json.load(f)["chars"]


def load_model(chars, checkpoint=None):
    model = CRNN(len(chars) + 1)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model", state))
    return model.eval()


def prepare(image):
    image = ImageOps.exif_transpose(image).convert("L")
    if sum(image.getdata()) / (image.width * image.height) < 128:
        image = ImageOps.invert(image)
    width = max(STRIDE, round(image.width * HEIGHT / image.height))
    image, _ = to_reading_order(image.resize((width, HEIGHT), Image.LANCZOS), "")
    return image


def to_tensor(image):
    pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return pixels.float().div(255).view(1, 1, image.height, image.width)


def read(image, model, chars):
    with torch.no_grad():
        scores = model(to_tensor(prepare(image)))
    return from_reading_order(greedy_decode(scores.permute(1, 0, 2).numpy(), chars)[0])


def main():
    parser = argparse.ArgumentParser(description="Read the Urdu text in line images.")
    parser.add_argument("images", nargs="*")
    parser.add_argument("--text", help="render this sentence and read it back")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    if not args.images and not args.text:
        parser.error("give image paths or --text")

    chars = load_chars()
    model = load_model(chars, args.checkpoint)
    if not args.checkpoint:
        print("no --checkpoint, the model is untrained\n")

    for path in args.images:
        print(path)
        print(f"  read:     {read(Image.open(path), model, chars)}")

    if args.text:
        image = render_line(args.text, load_font())
        if args.augment:
            image = augment(image, rng_for(args.seed, 0))
        OUT.mkdir(exist_ok=True)
        image.save(OUT / "predict_input.png")
        print(OUT / "predict_input.png")
        print(f"  expected: {args.text}")
        print(f"  read:     {read(image, model, chars)}")


if __name__ == "__main__":
    main()
