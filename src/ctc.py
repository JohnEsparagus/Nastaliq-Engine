import numpy as np

BLANK = 0


def best_path(scores):
    letter = scores[..., 1:].argmax(axis=-1) + 1
    best_letter = np.take_along_axis(scores, letter[..., None], axis=-1)[..., 0]
    return np.where(scores[..., BLANK] > best_letter, BLANK, letter)


def emit_mask(path, lengths=None):
    previous = np.empty_like(path)
    previous[:, 0] = -1
    previous[:, 1:] = path[:, :-1]
    keep = (path != BLANK) & (path != previous)
    if lengths is not None:
        keep &= np.arange(path.shape[1]) < np.asarray(lengths)[:, None]
    return keep


def greedy_decode(scores, chars, lengths=None):
    scores = np.asarray(scores)
    if scores.ndim == 2:
        scores = scores[None]
    path = best_path(scores)
    keep = emit_mask(path, lengths)
    table = np.array([""] + list(chars), dtype=object)
    return ["".join(table[row[mask]]) for row, mask in zip(path, keep)]
