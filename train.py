import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from tokenizers import ByteLevelBPETokenizer

import math

from model import MyDataset, Embedder, Transformer


vocab_size = 20000
embedding_size = 512
num_heads = 8
num_blocks = 6
context_window = 128
ff_hidden_size = 1024

emb = Embedder(vocab_size, embedding_size, context_window)
text = 'this is a test segment of a text'
tokenizer = ByteLevelBPETokenizer.from_file('./models/tokenizer/vocab.json', './models/tokenizer/merges.txt')

tokens = tokenizer.encode(text)
token_texts = tokens.tokens
token_ids = torch.tensor(tokens.ids)
print(token_texts, token_ids)

transformer = Transformer(num_blocks, embedding_size, num_heads, ff_hidden_size, vocab_size, context_window)
logits = transformer(token_ids)
print(logits, logits.shape)
print(logits[-1])
print(torch.argmax(logits[-1]))
