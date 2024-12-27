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
    
    def get_dict_size(self):
        return len(self.chars)


# tokenizer = ByteLevelBPETokenizer()
# tokenizer.train('./data/summaries.txt', vocab_size=700, special_tokens=['<PAD>', '<MASK>'])

# text = ' Haggai\'s message is filled with an urgency for the people to proceed with the rebuilding of the second Jerusalem temple. Haggai attributes a recent drought to the peoples\' refusal to rebuild the temple, which he sees as key to Jerusalemâ€™s glory. The book ends with the prediction of the downfall of kingdoms, with one Zerubbabel, governor of Judah, as the Lordâ€™s chosen leader. The language here is not as finely wrought as in some other books of the minor prophets, yet the intent seems straightforward.'
# print(tokenizer.encode(text).tokens)

# with open('./data/summaries.txt', 'r', encoding='utf-8') as f:
#     data = list(f.read())
#     print(tokenizer.encode(data).tokens)

# tokenizer.save_model('./models/tokenizer')

# tokenizer = ByteLevelBPETokenizer.from_file('./models/tokenizer/vocab.json', './models/tokenizer/merges.txt')
