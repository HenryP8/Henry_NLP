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
        self.w_q = nn.Linear(d_embed, d_embed)
        self.w_k = nn.Linear(d_embed, d_embed)
        self.w_v = nn.Linear(d_embed, d_embed)
        self.w_o = nn.Linear(d_embed, d_embed)

        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_k = d_embed / n_heads

    def split(self, Q, K, V):
        Q_i = torch.split(Q, int(self.d_k), 1)
        K_i = torch.split(K, int(self.d_k), 1)
        V_i = torch.split(V, int(self.d_k), 1)

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

        Q_i, K_i, V_i = self.split(Q, K, V)

        QKV_i = [self.scaled_dot_product(Q_i[i], K_i[i], V_i[i]) for i in range(len(Q_i))]

        QKV = self.w_o(self.concat(QKV_i))

        return QKV


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()

    def forward(self, x):
        pass


class Embedder(nn.Module):
    def __init__(self, dict_size, d_embed):
        super(FeedForward, self).__init__()

        self.embed = nn.Embedding(dict_size, d_embed)

    def forward(self, x):
        x = self.embed(x)

        return x


att = Attention(6, 2)
x = torch.tensor([[.1, .6, .9, .2, .3, .8],
                  [.2, .7, .3, .5, .1, .3],
                  [.8, .9, .7, .6, .1, .2]])
print(att(x))

emb = Embedder(20000, 512)
text = 'this is a test segment of text'