import os
import pickle as pkl

import pandas as pd

from tqdm import tqdm

from tokenizers import ByteLevelBPETokenizer
    

class CharacterTokenizer():
    def __init__(self, data):
        self.chars = sorted(list(set(data)))
        self.str_to_int = {c:i for i,c in enumerate(self.chars)}
        self.int_to_str = {i:c for i,c in enumerate(self.chars)}

    def encode(self, chars):
        return [self.str_to_int[c] for c in chars]

    def decode(self, nums):
        return [self.int_to_str[n] for n in nums]
    
    def get_vocab_size(self):
        return len(self.chars)
    
    def get_tokenizer(self):
        with open('./models/tokenizer/character_tokenizer.pkl', 'rb') as f:
            tokenizer = pkl.load(f)
        return tokenizer


class BPETokenizer():
    def __init__(self, fn, vocab_size):
        self.fn = fn
        self.vocab_size = vocab_size
        self.tokenizer = self.get_tokenizer()

    def train(self):
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(self.fn, vocab_size=self.vocab_size, special_tokens=['<PAD>', '<MASK>'])

        dir_path = f'./models/{self.vocab_size}_BPE_tokenizer'
        if os.path.isdir(dir_path):
            tokenizer.save_model(dir_path)
        else:
            os.mkdir(dir_path)
            tokenizer.save_model(dir_path)

    def get_tokenizer(self):
        tokenizer = ByteLevelBPETokenizer.from_file(f'./models/{self.vocab_size}_BPE_tokenizer/vocab.json', 
                                                    f'./models/{self.vocab_size}_BPE_tokenizer/merges.txt')
        return tokenizer
    
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()
    
    def encode(self, x):
        return self.tokenizer.encode(x).ids

    def decode(self, y):
        return self.tokenizer.decode(y)