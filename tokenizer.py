import pandas as pd

from tqdm import tqdm

from tokenizers import ByteLevelBPETokenizer
    
# data = pd.read_pickle('./data/book_summary.pkl')
# train_data = data['summary'].to_numpy()

# with open('./data/summaries.txt', 'w', encoding='utf-8') as f:
#     for summary in train_data:
#         f.write(summary)
# f.close()

tokenizer = ByteLevelBPETokenizer()
tokenizer.train('./data/summaries.txt', vocab_size=20000, special_tokens=['<PAD>', '<MASK>'])

tokenizer.save_model('./models/tokenizer')

tokenizer = ByteLevelBPETokenizer.from_file('./models/tokenizer/vocab.json', './models/tokenizer/merges.txt')
