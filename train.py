import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from tokenizers import ByteLevelBPETokenizer

import pandas as pd
import numpy as np
from tqdm import tqdm

import math

from model import MyDataset, Embedder, Transformer


vocab_size = 20000
embedding_size = 512
num_heads = 8
num_blocks = 6
context_window = 100
ff_hidden_size = 1024
batch_size = 1

device = 'cuda'

df = pd.read_pickle('./data/book_summary.pkl')
train_data = df['summary'].to_numpy()
data = DataLoader(train_data, batch_size=batch_size, shuffle=True)

tokenizer = ByteLevelBPETokenizer.from_file('./models/tokenizer/vocab.json', './models/tokenizer/merges.txt')
transformer = Transformer(num_blocks, embedding_size, num_heads, ff_hidden_size, vocab_size, context_window, device).to(device)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0000001)
loss_fn = nn.CrossEntropyLoss()

for idx, text in enumerate(tqdm(data)):
    tokens = torch.tensor(tokenizer.encode(text[0]).ids).to(device)

    if len(tokens) <= context_window:
        continue

    next_token = tokens[context_window]
    tokens = tokens[:context_window]
    logits = transformer(tokens)

    pred = logits[-1]

    target = torch.zeros(20000).to(device)
    target[int(next_token)] = 1

    loss = loss_fn(pred, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
