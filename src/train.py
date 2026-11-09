import argparse
import time

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .cache import load_index
from .config import ROOT
from .ctc import greedy_decode
from .dataset import LineDataset, pad_collate
from .model import CRNN, num_classes
from .predict import load_chars


def main():
    rows = load_index()[:16]
    alphabet = load_chars()
    print(f"lines:     {len(rows)}, e.g. {rows[0][2][:50]}...")
    print(f"alphabet:  {len(alphabet)} characters + blank = {len(alphabet) + 1} classes\n")

    loader = DataLoader(LineDataset(rows), batch_size=16, collate_fn=pad_collate)
    images, targets, input_lengths, target_lengths = next(iter(loader))
    print("one batch from pad_collate:")
    print(f"  images          {tuple(images.shape)}   (N, 1, H, widest W)")
    print(f"  targets         {tuple(targets.shape)}   all 16 labels joined, ids 1..{len(alphabet)}")
    print(f"  input_lengths   {input_lengths.tolist()}   frames per line (width // 4)")
    print(f"  target_lengths  {target_lengths.tolist()}   characters per line\n")

    model = CRNN(num_classes())
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-3)
    ctc = nn.CTCLoss(blank=0)
    print(f"model:     {sum(p.numel() for p in model.parameters()):,} weights\n")

    log_probs = model(images)
    loss = ctc(log_probs, targets, input_lengths, target_lengths)
    print("one forward pass:")
    print(f"  log_probs       {tuple(log_probs.shape)}   (T, N, classes)")
    print(f"  every line fits: {bool((input_lengths <= log_probs.shape[0]).all())}   input_lengths <= T")
    print(f"  loss at step 0: {loss.item():.3f}")


if __name__ == "__main__":
    main()
