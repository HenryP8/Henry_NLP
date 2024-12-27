import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from tokenizers import ByteLevelBPETokenizer

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from tqdm import tqdm

import math
import random
import time

from model import Transformer
from tokenizer import CharacterTokenizer


with open('./data/summaries.txt', 'r', encoding='utf-8') as f:
    words = list(f.read())
tokenizer = CharacterTokenizer(words)


context_window = 128
dict_size = tokenizer.get_dict_size()
n_blocks = 6
d_embed = 512
n_heads = 8
hidden = 2048
device = 'cuda'

model = Transformer(n_blocks, d_embed, n_heads, hidden, dict_size, device).to(device)
model.load_state_dict(torch.load('./models/transformer/1735253654.015008.pth', weights_only=True))
model.eval()

tokens = torch.tensor([[0]]).to(device)

for _ in tqdm(range(500)):
    pred = model(tokens[:, -context_window:])
    pred = pred[:, -1, :]

    probs = F.softmax(pred, dim=1)

    next_token = torch.multinomial(probs, num_samples=1)

    tokens = torch.cat((tokens, next_token), dim=1)

tokens = tokens.detach().cpu().numpy()[0]
print(''.join(tokenizer.decode(tokens)))
