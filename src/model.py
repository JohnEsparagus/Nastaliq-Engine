import json
import math

import torch
from torch import nn

from .config import CHARSET
from .sequence import HEIGHT, STRIDE


def num_classes(path=CHARSET):
    with open(path, encoding="utf-8") as f:
        return len(json.load(f)["chars"]) + 1


def conv_block(c_in, c_out):
    return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    def __init__(self, classes, hidden=256):
        super().__init__()
        assert STRIDE == 4

        layers = conv_block(1, 64) + [nn.MaxPool2d(2)]
        layers += conv_block(64, 128) + [nn.MaxPool2d(2)]

        c_in = 128
        height_pools = int(math.log2(HEIGHT // 4))
        for c_out in [256, 256, 512, 512, 512, 512][:height_pools]:
            layers += conv_block(c_in, c_out) + [nn.MaxPool2d((2, 1))]
            c_in = c_out
        self.cnn = nn.Sequential(*layers)

        self.rnn = nn.LSTM(c_in, hidden, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(2 * hidden, classes)

    def forward(self, x):
        x = self.cnn(x).squeeze(2).permute(2, 0, 1)
        x, _ = self.rnn(x)
        return self.fc(x).log_softmax(-1)


def main():
    model = CRNN(num_classes())
    print(model)
    for name, part in [("cnn", model.cnn), ("rnn", model.rnn), ("fc", model.fc)]:
        print(f"{name:4} {sum(p.numel() for p in part.parameters()):>11,}")
    print(f"total {sum(p.numel() for p in model.parameters()):>10,}")
    with torch.no_grad():
        print("(2, 1, 64, 400) ->", tuple(model.eval()(torch.zeros(2, 1, HEIGHT, 400)).shape))


if __name__ == "__main__":
    main()
