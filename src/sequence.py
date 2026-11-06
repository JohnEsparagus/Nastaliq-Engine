import numpy as np
from bidi import get_display
from PIL import ImageOps

HEIGHT = 64
STRIDE = 4
DIRECTION = "reverse_label"

MIRROR = str.maketrans("()", ")(")


def required_frames(labels):
    single = isinstance(labels, str)
    if single:
        labels = [labels]

    lengths = np.array([len(label) for label in labels], dtype=np.int64)
    codes = np.frombuffer("".join(labels).encode("utf-32-le"), dtype=np.uint32)
    owner = np.repeat(np.arange(len(labels)), lengths)

    repeat = (codes[1:] == codes[:-1]) & (owner[1:] == owner[:-1])
    frames = lengths + np.bincount(owner[1:][repeat], minlength=len(labels))
    return int(frames[0]) if single else frames


def output_frames(width, stride=STRIDE):
    return np.asarray(width) // stride


def visual_order(label):
    return get_display(label, base_dir="R").translate(MIRROR)


def logical_order(visual):
    return get_display(visual.translate(MIRROR), base_dir="R")


def to_reading_order(image, label):
    visual = visual_order(label)
    if DIRECTION == "flip_image":
        return ImageOps.mirror(image), visual[::-1]
    return image, visual


def from_reading_order(text):
    if DIRECTION == "flip_image":
        text = text[::-1]
    return logical_order(text)
