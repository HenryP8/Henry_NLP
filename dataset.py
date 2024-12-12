from torch.utils.data import Dataset, DataLoader
import pandas as pd


# with open('./data/booksummaries.txt', errors="ignore") as f:
#     lines = f.readlines()
#
# titles = [l.split('\t')[2] for l in lines]
# summaries = [l.split('\t')[-1] for l in lines]
#
# df = pd.DataFrame({'title': titles, 'summary': summaries})
#
# print(df)
#
# df.to_pickle('./data/book_summary.pkl')


class MyDataset(Dataset):
    def __init__(self, df):
        self.title = df['title']
        self.summary = df['summary']

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        return self.summary.iloc[idx], self.title.iloc[idx]


df = pd.read_pickle('./data/book_summary.pkl')
ds = MyDataset(df)

dl = DataLoader(ds, batch_size=2, shuffle=True, drop_last=True)