import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from tqdm import tqdm

import math
import random
import time
import pickle as pkl

from model import Transformer
from tokenizer import CharacterTokenizer, BPETokenizer, PretrainedTokenizer


def get_data(data, batch_size, context_window):
    sample_points = torch.randint(1, len(data)-context_window-1, (batch_size,))

    samples = [torch.tensor(data[sp:sp+context_window]) for sp in sample_points]
    targets = [torch.tensor(data[sp+1:sp+context_window+1]) for sp in sample_points]
    x = torch.stack(samples)
    y = torch.stack(targets)

    return x, y


# tokenizer = BPETokenizer('./data/summaries.txt', 700)
# data = np.load('./data/token/PBE_tokenizer.npy')

tokenizer = PretrainedTokenizer()
data = np.load('./data/token/Pretrained_tokenizer.npy')

num_epochs = 15000
lr = 3e-4
context_window = 128
batch_size = 64
dict_size = tokenizer.get_vocab_size()

n_blocks = 8
d_embed = 512
n_heads = 8
hidden = 2048
device = 'cuda'

model = Transformer(n_blocks, d_embed, n_heads, hidden, dict_size, device).to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

losses = []

for i in tqdm(range(num_epochs)):

    train, target = get_data(data, batch_size, context_window)
    train, target = train.to(device), target.to(device)

    pred = model(train)

    loss = loss_fn(pred.view(batch_size*context_window, -1), target.view(batch_size*context_window))

    if i % 250 == 0 or num_epochs-i <= 1:
        print()
        print(i, loss.item())

    losses.append(loss.item())

    optim.zero_grad()
    loss.backward()
    optim.step()

cur_t = time.time()
torch.save(model.state_dict(), f'./models/transformer/{cur_t}.pth')
np.save(f'models/loss/{cur_t}.npy', np.array(losses))
