# profile_model.py
import time, torch
from src.model import CRNN

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CRNN(74).to(device).eval()

def timed(fn, x, reps=50):
    with torch.no_grad():
        for _ in range(5): fn(x)                 # warm-up
        if device == "cuda": torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps): fn(x)
        if device == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000  # ms

for batch in [1, 32]:
    for width in [200, 400, 800]:
        x = torch.randn(batch, 1, 64, width, device=device)
        feats = model.cnn(x).squeeze(2).permute(2, 0, 1)
        t_cnn = timed(model.cnn, x)
        t_rnn = timed(lambda f: model.rnn(f), feats)
        print(f"batch {batch:2d} width {width}: "
              f"CNN {t_cnn:6.2f} ms  LSTM {t_rnn:6.2f} ms  "
              f"LSTM share {t_rnn / (t_cnn + t_rnn):.0%}")