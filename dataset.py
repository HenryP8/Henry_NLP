import pandas as pd

with open('./data/booksummaries.txt', errors="ignore") as f:
    lines = f.readlines()

titles = [l.split('\t')[2] for l in lines]
summaries = [l.split('\t')[-1] for l in lines]

df = pd.DataFrame({'title': titles, 'summary': summaries})

print(df)

df.to_pickle('./data/book_summary.pkl')
