from dataclasses import dataclass
import torch, os

@dataclass
class CFG:
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert_model = 'bert-base-uncased'
    max_text_len = 128
    audio_sr = 16000
    n_mfcc = 40
    physio_dim = 10
    text_emb_dim = 768
    audio_emb_dim = 128
    physio_emb_dim = 64
    lstm_hidden = 256
    num_classes = 2
    lr = 1e-4
    batch_size = 16
    epochs = 30
    dropout = 0.3
    ckpt_dir = './checkpoints'
    task = 'classification'
os.makedirs(CFG.ckpt_dir, exist_ok=True)
