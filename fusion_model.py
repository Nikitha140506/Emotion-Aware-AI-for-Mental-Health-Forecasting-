import torch
import torch.nn as nn

class BiLSTMAttentionFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=2, dropout=0.3, task='classification'):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.att = nn.Linear(2*hidden_dim, 1)
        self.fc = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes if task=='classification' else 1)
        )
        self.task = task

    def forward(self, x):
        out, _ = self.lstm(x)
        attn = torch.softmax(self.att(out), dim=1)
        context = torch.sum(out * attn, dim=1)
        logits = self.fc(context)
        return logits, attn
