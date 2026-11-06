import io
import math
import random
from statistics import NormalDist

import numpy as np
from PIL import Image, ImageChops, ImageFilter

from .render import HEIGHT

RANGES = {
    "rotate_deg": 1.5,
    "rotate_drift": 0.3,
    "corner_jitter": 0.06,
    "paper": (175, 250),
    "paper_blotch": 12,
    "ink": (0, 70),
    "lighting": 0.3,
    "min_contrast": 70,
    "blur_sigma": 0.012,
    "downscale": (0.65, 1.0),
    "noise_sigma": (2, 8),
    "jpeg_quality": (40, 90),
}
PROBABILITY = {"lighting": 0.7, "blur": 0.5, "downscale": 0.3, "noise": 0.7, "jpeg": 0.6}

NORMAL_QUANTILES = [NormalDist().inv_cdf((i + 0.5) / 256) for i in range(256)]


def rng_for(seed, index):
    return random.Random(f"{seed}-{index}")


def perspective_coefficients(output_corners, input_corners):
    a, b = [], []
    for (x, y), (u, v) in zip(output_corners, input_corners):
        a.append([x, y, 1, 0, 0, 0, -x * u, -y * u])
        a.append([0, 0, 0, x, y, 1, -x * v, -y * v])
        b += [u, v]
    return np.linalg.solve(a, b).tolist()


def max_angle(width, height):
    drift = math.degrees(math.atan(RANGES["rotate_drift"] * height / width))
    return min(RANGES["rotate_deg"], drift)


def geometry(clean, angle_deg, corner_offsets):
    w, h = clean.size
    pad = h // 2
    canvas = Image.new("L", (w + 2 * pad, h + 2 * pad), 255)
    canvas.paste(clean, (pad, pad))

    cx, cy = pad + w / 2, pad + h / 2
    corners = [(pad, pad), (pad + w, pad), (pad + w, pad + h), (pad, pad + h)]
    a = math.radians(angle_deg)
    moved = []
    for (x, y), (dx, dy) in zip(corners, corner_offsets):
        rx = cx + (x - cx) * math.cos(a) - (y - cy) * math.sin(a)
        ry = cy + (x - cx) * math.sin(a) + (y - cy) * math.cos(a)
        moved.append((rx + dx, ry + dy))
    coefficients = perspective_coefficients(moved, corners)
    warped = canvas.transform(canvas.size, Image.PERSPECTIVE, coefficients, Image.BICUBIC, fillcolor=255)

    ink = ImageChops.invert(warped).point(lambda v: 255 if v > 8 else 0).getbbox()
    if ink is None:
        raise ValueError("geometry lost all the ink")
    margin = round(0.1 * h)
    left, right = max(ink[0] - margin, 0), min(ink[2] + margin, warped.width)
    edge = max(2, round(0.03 * h))
    top = max(min(ink[1] - edge, cy - h / 2), 0)
    bottom = min(max(ink[3] + edge, cy + h / 2), warped.height)
    cropped = warped.crop((left, round(top), right, round(bottom)))
    return cropped.resize((max(1, round(cropped.width * HEIGHT / cropped.height)), HEIGHT), Image.LANCZOS)


def touches_border(clean):
    ink = ImageChops.invert(clean).point(lambda v: 255 if v > 8 else 0).getbbox()
    return ink is None or ink[0] == 0 or ink[1] == 0 or ink[2] == clean.width or ink[3] == clean.height


def smooth_field(rng, size, cells_per_height, low, high):
    w, h = size
    gw, gh = max(2, round(cells_per_height * w / h)), cells_per_height
    grid = Image.new("L", (gw, gh))
    grid.putdata([round(rng.uniform(low, high)) for _ in range(gw * gh)])
    return grid.resize(size, Image.BICUBIC)


def gaussian_noise(rng, size, sigma):
    uniform = Image.frombytes("L", size, rng.randbytes(size[0] * size[1]))
    return uniform.point([min(255, max(0, round(128 + sigma * z))) for z in NORMAL_QUANTILES])


def augment(clean, rng):
    r, p = RANGES, PROBABILITY
    w, h = clean.size

    limit = max_angle(w, h)
    jitter = r["corner_jitter"] * h
    image = geometry(
        clean,
        rng.uniform(-limit, limit),
        [(rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)) for _ in range(4)],
    )
    size = image.size

    paper_level = rng.uniform(*r["paper"])
    paper = smooth_field(rng, size, 1, paper_level - r["paper_blotch"], paper_level + r["paper_blotch"])
    ink_level = rng.uniform(*r["ink"])
    image = Image.composite(Image.new("L", size, round(ink_level)), paper, ImageChops.invert(image))

    if rng.random() < p["lighting"]:
        darkest = paper_level - r["paper_blotch"]
        allowed = max(0.0, 1 - (ink_level + r["min_contrast"]) / darkest)
        strength = rng.uniform(0, min(r["lighting"], allowed))
        light = smooth_field(rng, size, 1, round(255 * (1 - strength)), 255)
        image = ImageChops.multiply(image, light)

    if rng.random() < p["blur"]:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, r["blur_sigma"] * HEIGHT)))

    if rng.random() < p["downscale"]:
        f = rng.uniform(*r["downscale"])
        small = image.resize((max(1, round(size[0] * f)), max(1, round(size[1] * f))), Image.BILINEAR)
        image = small.resize(size, Image.BILINEAR)

    if rng.random() < p["noise"]:
        image = ImageChops.add(image, gaussian_noise(rng, size, rng.uniform(*r["noise_sigma"])), 1.0, -128)

    if rng.random() < p["jpeg"]:
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=rng.randint(*r["jpeg_quality"]))
        image = Image.open(io.BytesIO(buffer.getvalue())).convert("L")

    return image
