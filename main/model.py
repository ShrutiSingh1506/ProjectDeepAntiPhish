# model.py

import torch
import torch.nn as nn
from typing import Sequence, Union
from torch.utils.data import Dataset

class SparseRowDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row_dense = self.X[idx].toarray().ravel()
        return torch.from_numpy(row_dense).float(), self.y[idx]

class DeepAntiPhish(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: Sequence[int] = (2048, 1024, 512, 128, 64, 16, 8, 1),
        drop_probs: Union[Sequence[float], float] = (.3, .2, .2, .1, .05, .02, 0),
    ):
        super().__init__()

        if isinstance(drop_probs, (float, int)):
            drop_probs = [float(drop_probs)] * len(hidden_dims)

        layers = []
        prev = input_dims
        for size_hidden, prob in zip(hidden_dims, drop_probs):
            layers += [
                nn.Linear(prev, size_hidden, bias=True),
                nn.BatchNorm1d(size_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(prob),
            ]
            prev = size_hidden

        layers.extend([
            nn.Linear(prev, hidden_dims[-1], bias=True),
            nn.Dropout(drop_probs[-1]),
        ])
        self.net = nn.Sequential(*layers)
        self._init_weights()
        
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x.toarray()).float()
            x = x.to(next(self.parameters()).device, non_blocking=True)

        return self.net(x).squeeze(1)