import random
import unittest

import numpy as np

from src.ctc import greedy_decode

CHARS = ["a", "b"]


def frames(pattern, chars=CHARS):
    classes = len(chars) + 1
    table = []
    for symbol in pattern:
        hot = 0 if symbol == "-" else chars.index(symbol) + 1
        table.append([0.9 if c == hot else 0.1 / (classes - 1) for c in range(classes)])
    return table


def slow_decode(probs, chars):
    out, previous = [], None
    for frame in probs:
        best = max(range(len(frame)), key=lambda i: (frame[i], i > 0))
        if best != 0 and best != previous:
            out.append(chars[best - 1])
        previous = best
    return "".join(out)


class TestGreedyDecode(unittest.TestCase):
    def test_plan_example(self):
        self.assertEqual(greedy_decode(frames("aa-abb-"), CHARS), ["aab"])

    def test_all_blank(self):
        self.assertEqual(greedy_decode(frames("----"), CHARS), [""])

    def test_repeated_character(self):
        self.assertEqual(greedy_decode(frames("aaaa"), CHARS), ["a"])
        self.assertEqual(greedy_decode(frames("a-a"), CHARS), ["aa"])
        self.assertEqual(greedy_decode(frames("aa--aa"), CHARS), ["aa"])

    def test_single_frame(self):
        self.assertEqual(greedy_decode(frames("a"), CHARS), ["a"])
        self.assertEqual(greedy_decode(frames("-"), CHARS), [""])

    def test_matches_slow_decode_including_ties(self):
        rng = random.Random(0)
        chars = [chr(0x0627 + i) for i in range(6)]
        for _ in range(300):
            table = [[round(rng.random(), 1) for _ in range(len(chars) + 1)] for _ in range(rng.randint(1, 40))]
            self.assertEqual(greedy_decode(table, chars), [slow_decode(table, chars)])

    def test_padding_is_ignored(self):
        batch = np.array([frames("aa-abb-"), frames("a-a") + frames("bbbb")])
        self.assertEqual(greedy_decode(batch, CHARS, lengths=[7, 3]), ["aab", "aa"])

    def test_logits_match_probs(self):
        probs = np.array(frames("ab-ba"))
        self.assertEqual(greedy_decode(np.log(probs), CHARS), greedy_decode(probs, CHARS))


if __name__ == "__main__":
    unittest.main()
