import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torchtext.data import Field
from torchtext.data import Dataset, Example
from torchtext.data import BucketIterator
from torchtext.vocab import FastText
from torchtext.vocab import CharNGram

import torchtext

import numpy as np
import pandas as pd

def train_fasttext(train_dataset, test_dataset):
    data_dict = {"text":[], "label":[]}
    for i in range(len(train_dataset)):
        data_dict['text'].append(train_dataset.data[i])
        data_dict['label'].append(train_dataset.labels[i])

    for i in range(len(test_dataset)):
        data_dict['text'].append(test_dataset.data[i])
        data_dict['label'].append(test_dataset.labels[i])

    df = pd.DataFrame.from_dict(data_dict, orient='columns')

    text_field = Field(
        tokenize='basic_english',
        lower=True
    )
    label_field = Field(sequential=False, use_vocab=False)
    # sadly have to apply preprocess manually
    preprocessed_text = df['text'].apply(lambda x: text_field.preprocess(x))
    # load fastext simple embedding with 300d
    text_field.build_vocab(
        preprocessed_text,
        vectors='fasttext.simple.300d'
    )
    # get the vocab instance
    vocab = text_field.vocab

    from torchtext.vocab import FastText
    embedding = FastText('simple')

    from torchtext.vocab import CharNGram
    embedding_charngram = CharNGram()

    from torchtext.vocab import GloVe
    embedding_glove = GloVe(name='6B', dim=100)

    # known token, in my case print 12
    print(vocab['are'])
    # unknown token, will print 0
    print(vocab['crazy'])

    from torchtext.data import Dataset, Example
    #ltoi = {l: i for i, l in enumerate(df['label'].unique())}
    #df['label'] = df['label'].apply(lambda y: ltoi[y])

    class DataFrameDataset(Dataset):
        def __init__(self, df: pd.DataFrame, fields: tuple):
            super(DataFrameDataset, self).__init__(
                [
                    Example.fromlist(list(r), fields)
                    for i, r in df.iterrows()
                ],
                fields
            )

    train_dataset, test_dataset = DataFrameDataset(
        df=df,
        fields=(
            ('text', text_field),
            ('label', label_field)
        )
    ).split()

    from torchtext.data import BucketIterator
    train_iter, test_iter = BucketIterator.splits(
        datasets=(train_dataset, test_dataset),
        batch_sizes=(2, 2),
        sort=False
    )

    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    class ModelParam(object):
        def __init__(self, param_dict: dict = dict()):
            self.input_size = param_dict.get('input_size', 0)
            self.vocab_size = param_dict.get('vocab_size')
            self.embedding_dim = param_dict.get('embedding_dim', 300)
            self.target_dim = param_dict.get('target_dim', 2)

    class MyModel(nn.Module):
        def __init__(self, model_param: ModelParam):
            super().__init__()
            self.embedding = nn.Embedding(
                model_param.vocab_size,
                model_param.embedding_dim
            )
            self.lin = nn.Linear(
                model_param.input_size * model_param.embedding_dim,
                model_param.target_dim
            )

        def forward(self, x):
            features = self.embedding(x).view(x.size()[0], -1)
            features = F.relu(features)
            features = self.lin(features)
            return features

    model_param = ModelParam(
        param_dict=dict(
            vocab_size=len(text_field.vocab),
            input_size=5
        )
    )
    model = MyModel(model_param)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    epochs = 5

    for epoch in range(epochs):
        epoch_losses = list()
        for batch in train_iter:
            optimizer.zero_grad()

            print(type(batch.text.T))
            print(len(batch.text.T))
            print(type(batch.text.T[0]))
            print(batch.text.T.shape)
            prediction = model(batch.text.T)

            loss = loss_function(prediction, batch.label)

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
        print('train loss on epoch {} : {:.3f}'.format(epoch,
                                                       np.mean(epoch_losses)))

        test_losses = list()
        for batch in test_iter:
            with torch.no_grad():
                optimizer.zero_grad()
                prediction = model(batch.text.T)
                loss = loss_function(prediction, batch.label)

                test_losses.append(loss.item())

        print(
            'test loss on epoch {}: {:.3f}'.format(epoch, np.mean(test_losses)))

