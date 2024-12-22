import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from tokenizers import ByteLevelBPETokenizer

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

        self.norm = nn.LayerNorm(d_embed)

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
        masked_QK = self.mask(QK)
        scaled_QK = masked_QK / math.sqrt(self.d_k)
        scaled_QK = torch.softmax(scaled_QK, dim=1)
        QKV = torch.matmul(scaled_QK, V)

        return QKV

    def mask(self,A):
        mask = torch.tril(torch.ones(A.shape))
        masked = A.masked_fill(mask==0, float('-inf'))

        return masked

    def concat(self, res_i):
        ret = torch.cat(res_i, 1)
        
        return ret

    def forward(self, x):
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        Q_i, K_i, V_i = self.split(Q, K, V)

        QKV_i = [self.scaled_dot_product(Q_i[i], K_i[i], V_i[i]) for i in range(len(Q_i))]

        QKV = self.w_o(self.concat(QKV_i))

        add_norm = self.norm(QKV + x)

        return add_norm


class FeedForward(nn.Module):
    def __init__(self, d_embed, hidden):
        super(FeedForward, self).__init__()

        self.norm = nn.LayerNorm(d_embed)
        self.l1 = nn.Linear(d_embed, hidden)
        self.l2 = nn.Linear(hidden, d_embed)
        self.gelu = nn.GELU()

    def forward(self, x):
        ret = self.l1(x)
        ret = self.gelu(ret)
        ret = self.l2(ret)

        ret = self.norm(ret + x)

        return ret


class Embedder(nn.Module):
    def __init__(self, dict_size, d_embed, context_window):
        super(Embedder, self).__init__()

        self.context_window = context_window

        self.embed = nn.Embedding(dict_size, d_embed)
        self.pe = torch.zeros(context_window, d_embed)

        pos = torch.arange(context_window)
        pos = pos.view(context_window, -1)
        pos = pos.expand(context_window, int(d_embed / 2))

        odd_denom = torch.arange(1, d_embed, 2)
        odd_denom = torch.sub(odd_denom, 1)
        odd_denom = torch.div(odd_denom, d_embed)
        odd_denom = torch.pow(10000, odd_denom)

        even_denom = torch.arange(0, d_embed, 2)
        even_denom = torch.div(even_denom, d_embed)
        even_denom = torch.pow(10000, even_denom)

        self.pe[:, 0::2] = torch.sin(torch.div(pos, even_denom))
        self.pe[:, 1::2] = torch.cos(torch.div(pos, odd_denom))

    def forward(self, x):
        x = F.pad(x, (0, self.context_window - x.shape[0]), 'constant', 0)
        x = self.embed(x)

        return x + self.pe


vocab_size = 20000
embedding_size = 512
num_heads = 8
context_window = 128
ff_hidden_size = 1024

emb = Embedder(vocab_size, embedding_size, context_window)
text = 'this is a test segment of a text'
tokenizer = ByteLevelBPETokenizer.from_file('./models/tokenizer/vocab.json', './models/tokenizer/merges.txt')

tokens = tokenizer.encode(text)
token_texts = tokens.tokens
token_ids = torch.tensor(tokens.ids)
print(token_texts, token_ids)

embedding = emb(token_ids)
print(embedding, embedding.shape)

att = Attention(embedding_size, num_heads)
attention = att(embedding)
print(attention, attention.shape)

ff = FeedForward(embedding_size, ff_hidden_size)
res = ff(attention)
print(res, res.shape)
