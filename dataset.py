#%%
from webbrowser import get
import pandas as pd
from tqdm import tqdm
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn.functional as F

class TextDataset(torch.utils.data.Dataset):
    """Some Information about TextDataset"""
    def __init__(self, df):
        super(TextDataset, self).__init__()
        self.df = df

    def __getitem__(self, index):
        row = self.df.iloc[index]
        data = row['data']
        label = row['label']
        label = int(label)
        return [data, label]

    def __len__(self):
        return len(self.df)





def process_dataset(data, test=False):
    df = pd.DataFrame(data)
    df['text'] = df[0].apply(lambda x: x.strip('\n'))
    df = df.drop(columns=0)
    if test:
        df['data'] = df.text.apply(lambda x: x.split('\t')[1])
        df['label'] = df.text.apply(lambda x: x.split('\t')[0])
    else:
        df['data'] = df.text.apply(lambda x: x.split('\t')[0])
        df['label'] = df.text.apply(lambda x: x.split('\t')[1])
    df = df.drop(0)
    df = df.drop(columns=['text'])
    return df
def get_collate_fn(vectorizer):
    def vectorize_batch(batch):
        [X,Y] = list(zip(*batch))
        X = vectorizer.transform(X).todense()
        Y = torch.tensor(Y)
        Y = F.one_hot(Y, 2)

        return torch.tensor(X, dtype=torch.float32), Y.float()
    return vectorize_batch

def get_text_dataloaders(train_batch_size = 100, test_batch_size = None, topk=None):
    train_df, dev_df, vocab, vectorizer = init_data(topk = topk)
    if test_batch_size is None:
        test_batch_size = train_batch_size
    train_dataset = TextDataset(train_df)
    dev_dataset = TextDataset(dev_df)
    collate_fn = get_collate_fn(vectorizer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=collate_fn)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=test_batch_size, collate_fn=collate_fn)
    return [train_loader, dev_loader, vocab]
# %%
def init_data(topk = None):

    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)

    with open('data/SST-2/train.tsv') as f:
        train_data = f.readlines()

    with open('data/SST-2/dev.tsv') as f:
        dev_data = f.readlines()

    train_df = process_dataset(train_data)
    dev_df = process_dataset(dev_data)
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(train_df['data']), specials=["<unk>"])
    if topk is not None:
        vocabulary = vocab.get_itos()[:topk]
        vectorizer = CountVectorizer(vocabulary=vocabulary, tokenizer=tokenizer)
    else:
        vocabulary = vocab.get_itos()
        vectorizer = CountVectorizer(vocabulary=vocabulary, tokenizer=tokenizer)
    return train_df, dev_df, vocabulary, vectorizer