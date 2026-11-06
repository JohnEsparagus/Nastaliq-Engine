import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .augment import augment, rng_for
from .cache import load_clean, load_index
from .predict import load_chars, to_tensor
from .sequence import output_frames, to_reading_order


class LineDataset(Dataset):
    def __init__(self, rows=None, augment=False, seed=0):
        self.rows = load_index() if rows is None else rows
        self.augment = augment
        self.seed = seed
        self.char_to_id = {ch: i + 1 for i, ch in enumerate(load_chars())}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        index, _, label = self.rows[i]
        image = load_clean(index)
        if self.augment:
            image = augment(image, rng_for(self.seed, index))
        image, label = to_reading_order(image, label)
        ids = torch.tensor([self.char_to_id[ch] for ch in label], dtype=torch.long)
        return to_tensor(image)[0], ids


def pad_collate(batch):
    images, targets = zip(*batch)
    max_w = max(img.shape[-1] for img in images)

    padded_img = []
    for img in images:
        pad_amount = max_w - img.shape[-1]
        padded_img.append(F.pad(img, (0, pad_amount), value=1.0))

    input_lengths = torch.tensor([output_frames(img.shape[-1]) for img in images], dtype=torch.long)
    target_lengths = torch.tensor([len(ids) for ids in targets], dtype=torch.long)
    return torch.stack(padded_img), torch.cat(targets), input_lengths, target_lengths