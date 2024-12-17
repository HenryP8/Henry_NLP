import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import math


class MyDataset(Dataset):
    def __init__(self, df):
        self.title = df['title']
        self.summary = df['summary']

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        return self.summary.iloc[idx], self.title.iloc[idx]
    

class Attention(nn.Module):
    def __init__(self, d_embed, n_heads):
        super(Attention, self).__init__()
        self.w_q = 0
        self.w_k = 0
        self.w_v = 0

        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_k = d_embed / n_heads

    def split(self, Q, K, V):
        Q_i = torch.split(Q, 2, 1)
        K_i = torch.split(K, 2, 1)
        V_i = torch.split(V, 2, 1)

        return Q_i, K_i, V_i

    def scaled_dot_product(self, Q, K, V):
        QK = torch.matmul(Q, torch.transpose(K, 0, 1))
        scaled_QK = QK / math.sqrt(self.d_k)
        scaled_QK = torch.softmax(scaled_QK, dim=1)
        QKV = torch.matmul(scaled_QK, V)

        return QKV
    
    def concat(self, res_i):
        ret = res_i[0]
        for res in res_i[1:]:
            ret = torch.cat((ret, res), 1)

        return ret

    def forward(self, x):
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()

    def forward(x):
        pass


att = Attention(2, 2)
Q = torch.tensor([[.1,.2,.3,.5], [.4,.5,.6,.2]], dtype=torch.float)
K = torch.tensor([[.6,.3,.2,.3], [.1,.8,.9,.1]], dtype=torch.float)
V = torch.tensor([[.3,.6,.8,.7], [.1,.2,.5,.9]], dtype=torch.float)

# print(att.scaled_dot_product(Q, K, V))

Q_i, K_i, V_i = att.split(Q, K, V)
print(att.concat(Q_i))
