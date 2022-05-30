from read_data import read_folder_of_authors
from tweet_preprocessor import preprocess_tweet
import os
import emot
import pickle
from HSDataset import HSDataset
from fasttext import train_fasttext
import argparse
import torch
# import traducer

BI_FOLDER = "pan21-author-profiling-training-2021-03-14"
EN_FOLDER = "en"
ES_FOLDER = "es"
import torch.nn as nn

class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()

        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768, 512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, 2)

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        cls_hs = self.bert(sent_id, attention_mask=mask)[1]

        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x


def _parse_args():
    parser = argparse.ArgumentParser()

    # model and preprocessing args
    parser.add_argument("--model-name", type=str, default="Pretrained_huge_dataset_BERT")
    parser.add_argument("--preprocessing-method", type=str, default="classical")
    parser.add_argument("--load-preprocessed", type=str, default=False)
    parser.add_argument("--num-tweets-per-batch", type=int, default=100)
    """
    if num_tweets_per_batch is set to 100 then all the tweets of an author 
    will be concatenated and viewed as one, if is set to 1 then we will 
    consider each tweet individually, otherwise if it is between 2 and 99 
    a list containing a batch of randomly sampled tweets will be created
    """

    # hyperparams args
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-1)
    parser.add_argument("--backbone", type=str, default="")
    parser.add_argument("--max-seq-len", type=int, default=512)
    """
    backbone should be completed further if, for instance different arhitectures
    of the same model will be tried(e.g. model_name would be LSTM and the same 
    preprocessing might be needed, while the backbone of the LSTM may differ
    max_seq_len might be required in different word embeddings
    """

    # multi threading args
    parser.add_argument("--num-workers", type=int, default=1)
    """
    if setting the num_workers argument to something higher than 1 is desired
    in order to speed up the computation, the implementation of a "collate_fn"
    and a "worker_init_fn" will be needed
    """

    # data split args
    parser.add_argument("--train-data-pct", type=float, default=0.8)
    parser.add_argument("--val-data-pct", type=float, default=0.0)
    parser.add_argument("--test-data-pct", type=float, default=0.2)
    """
    val_data_pct should be kept to 0.0 until the volume of data increases 
    significantly
    """

    # supported languages args
    parser.add_argument("--langs", type=list, default=['en'])
    """
    the list of languages should be one of either: ['en'], ['es'], ['en', 'es'], 
    ['es', 'en']
    """
    return parser.parse_args()


def train_test_split_hsdataset(args):
    if args.langs == ['en']:
        train_dataset = HSDataset(os.path.join(BI_FOLDER, EN_FOLDER),
                                  "train", args=args)
        val_dataset = HSDataset(os.path.join(BI_FOLDER, EN_FOLDER),
                                  "val", args=args)
        test_dataset = HSDataset(os.path.join(BI_FOLDER, EN_FOLDER),
                                 "test", args=args)
    elif args.langs == ['es']:
        train_dataset = HSDataset(os.path.join(BI_FOLDER, ES_FOLDER),
                                  "train", args=args)
        val_dataset = HSDataset(os.path.join(BI_FOLDER, ES_FOLDER),
                                "val", args=args)
        test_dataset = HSDataset(os.path.join(BI_FOLDER, ES_FOLDER),
                                 "test", args=args)
    elif len(args.langs) == 2 and 'en' in args.langs and 'es' in args.langs:
        train_dataset = HSDataset(BI_FOLDER, "train", args=args)
        val_dataset = HSDataset(BI_FOLDER, "val", args=args)
        test_dataset = HSDataset(BI_FOLDER, "test", args=args)
    else:
        print("An invalid list of languages was given")
        return None, None, None
    return train_dataset, val_dataset, test_dataset


def main(args):
    traduced = False

    # if not traduced:
    #     traducer.double_data()

    train_dataset, _, test_dataset = train_test_split_hsdataset(args)

    # print('DATA: ', train_dataset.data)

    args = _parse_args()
    if args.model_name.startswith("BERT"):
        # from BERT import train_bert
        from BERT2 import train_bert
        if args.load_preprocessed:
            train_dataset = pickle.load(open("train_bert.pickle", 'rb'))
            test_dataset = pickle.load(open("test_bert.pickle", 'rb'))

            train_bert(train_dataset, test_dataset)
        else:
            for i in range(len(train_dataset.data)):
                train_dataset.data[i] = preprocess_tweet(
                    train_dataset.data[i],
                    preprocessing_method=args.preprocessing_method)
            train_store = open("train_bert_.pickle", "wb")
            pickle.dump(train_dataset, train_store)

            for i in range(len(test_dataset.data)):
                test_dataset.data[i] = preprocess_tweet(
                    test_dataset.data[i],
                    preprocessing_method=args.preprocessing_method)
            test_store = open("test_bert_.pickle", "wb")
            pickle.dump(test_dataset, test_store)
    
            train_dataset = HSDataset(os.path.join(BI_FOLDER, EN_FOLDER), "train")
            test_dataset = HSDataset(os.path.join(BI_FOLDER, EN_FOLDER), "test")

            train_bert(train_dataset, test_dataset)

    elif args.model_name == 'FastText':
        if args.load_preprocessed == False:
            for i in range(len(train_dataset.data)):
                train_dataset.data[i] = preprocess_tweet(
                    train_dataset.data[i],
                    preprocessing_method=args.preprocessing_method)
            print("train was read")
            for i in range  (len(test_dataset.data)):
                test_dataset.data[i] = preprocess_tweet(
                    test_dataset.data[i],
                    preprocessing_method=args.preprocessing_method)
            print("test was read")
            train_fasttext(train_dataset, test_dataset)
    elif args.model_name == "Pretrained_huge_dataset_BERT":
        path = "huge_hs_dataset_model.pt"
        from transformers import AutoModel, BertTokenizerFast
        bert = AutoModel.from_pretrained('bert-base-uncased')
        model = BERT_Arch(bert)
        model.load_state_dict(torch.load(path, map_location=torch.device(
            'cpu')))
        device = torch.device('cpu')

        train_seq = torch.load("bert_data/train_seq.pt")
        train_mask = torch.load("bert_data/train_mask.pt")
        train_y = torch.load("bert_data/train_y.pt")

        # val_mask = torch.load("bert_data/val_mask.pt")
        # val_y = torch.load("bert_data/val_y.pt")
        # val_seq = torch.load("bert_data/val_seq.pt", map_location=device)

        import numpy as np
        from sklearn.metrics import classification_report

        with torch.no_grad():
            preds = model(train_seq.to(device), train_mask.to(device))
            preds = preds.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
            print(classification_report(train_y, preds))


if __name__=='__main__':
    args = _parse_args()
    main(args)
