import pandas as pd

from tokenizers import ByteLevelBPETokenizer
    
# data = pd.read_pickle('./data/book_summary.pkl')
# train_data = data['summary'].to_numpy()

# with open('./data/summaries.txt', 'w', encoding='utf-8') as f:
#     for summary in train_data:
#         f.write(summary)
# f.close()

tokenizer = ByteLevelBPETokenizer()
tokenizer.train('./data/summaries.txt', vocab_size=20000)

print('here')
encoding = tokenizer.encode('The novel, set in Glasgow, revolves around the central character, Rilke, an auctioneer who has agreed to quickly process and sell an inventory of largely valuable contents belonging to a recently deceased old man in exchange for a considerable fee. While sorting through some of the possessions in an attic, he comes across a collection of violent and potentially snuff pornography that appears to document the death of a mysterious young woman. Starting with local pornography trade contacts, Rilke sets out to discover this woman\'s identity and uncover the story behind her appearance in the disturbing photographs.')
print(encoding.tokens)
