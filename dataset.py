import pandas as pd
import numpy as np
import pickle as pkl

from tokenizer import CharacterTokenizer
from tokenizers import ByteLevelBPETokenizer


with open('./data/summaries.txt', 'r', encoding='utf-8') as f:
    words = f.read()

# tokenizer = CharacterTokenizer(words)
# tokens = tokenizer.encode(words)
# with open('./models/tokenizer/character_tokenizer.pkl', 'wb') as f:
#     pkl.dump(tokenizer, f)

tokenizer = ByteLevelBPETokenizer.from_file('./models/tokenizer/vocab.json', './models/tokenizer/merges.txt')
tokens = tokenizer.encode(words)
ids = tokens.ids

np.save(f'./data/token/PBE_tokenizer.npy', np.array(ids))
