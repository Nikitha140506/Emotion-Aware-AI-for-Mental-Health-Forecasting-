import numpy as np
from torch.utils.data import Dataset
from config import CFG

class PlaceholderDataset(Dataset):
    def __init__(self, n_samples=200):
        self.n = n_samples

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        text = "i am feeling fine" if idx % 2 == 0 else "i feel sad and tired"
        audio = np.random.randn(int(CFG.audio_sr * 5)).astype(np.float32)
        physio = np.random.randn(CFG.physio_dim).astype(np.float32)
        label = 0 if idx % 2 == 0 else 1
        return {'text': text, 'audio': audio, 'physio': physio, 'label': label}
