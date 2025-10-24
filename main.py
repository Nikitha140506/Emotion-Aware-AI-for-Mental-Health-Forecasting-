import torch, argparse
from torch import nn
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizerFast, BertModel
import numpy as np
from tqdm import tqdm
from config import CFG
from data.dataset_placeholder import PlaceholderDataset
from models.fusion_model import BiLSTMAttentionFusion
from utils.seed_utils import set_seed
from utils.metrics import compute_metrics

set_seed(CFG.seed)

tokenizer = BertTokenizerFast.from_pretrained(CFG.bert_model)

def collate_fn(batch):
    texts = [b['text'] for b in batch]
    enc = tokenizer(texts, truncation=True, padding='max_length',
                    max_length=CFG.max_text_len, return_tensors='pt')
    input_ids = enc['input_ids']
    att_mask = enc['attention_mask']
    audio_feats = torch.tensor([np.mean(b['audio']) for b in batch]).unsqueeze(1).float()
    physio_feats = torch.tensor([b['physio'] for b in batch]).float()
    labels = torch.tensor([b['label'] for b in batch]).long()
    return input_ids, att_mask, audio_feats, physio_feats, labels

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(CFG.bert_model)
        self.fc_audio = nn.Linear(1, 64)
        self.fc_physio = nn.Linear(CFG.physio_dim, 64)
        self.fusion = BiLSTMAttentionFusion(768 + 64 + 64, CFG.lstm_hidden, CFG.num_classes)

    def forward(self, input_ids, att_mask, audio, physio):
        t = self.bert(input_ids=input_ids, attention_mask=att_mask).pooler_output
        a = torch.relu(self.fc_audio(audio))
        p = torch.relu(self.fc_physio(physio))
        fused = torch.cat([t, a, p], dim=1).unsqueeze(1)
        logits, att = self.fusion(fused)
        return logits

def train_model():
    dataset = PlaceholderDataset(300)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=8, collate_fn=collate_fn)

    model = SimpleModel().to(CFG.device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = -1
    for epoch in range(3):
        model.train()
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}"):
            ids, mask, aud, phy, lab = [b.to(CFG.device) for b in batch]
            opt.zero_grad()
            out = model(ids, mask, aud, phy)
            loss = loss_fn(out, lab)
            loss.backward()
            opt.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in val_dl:
                ids, mask, aud, phy, lab = [b.to(CFG.device) for b in batch]
                out = model(ids, mask, aud, phy)
                preds = torch.argmax(out, dim=1).cpu().numpy()
                y_true.extend(lab.cpu().numpy())
                y_pred.extend(preds)
        acc, f1, auc = compute_metrics(y_true, y_pred)
        print(f"Val Acc: {acc:.3f}, F1: {f1:.3f}, AUC: {auc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    if args.train:
        train_model()
    else:
        print("Use --train to start training.")
