from read_data import read_folder_of_authors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, Sampler, Subset


class HSDataset(Dataset):
    def __init__(self, path, dataset_type, args):
        print('PATH:', path)
        self.path = path
        self.lang = "en" if "en" in self.path else "es"
        self.dataset_type = dataset_type
        self.args = args

        if self.dataset_type == 'train' and args.train_data_pct > 0.0:
            self.data, self.labels = read_folder_of_authors(self.path,
                                                            self.args,
                                                            self.dataset_type)
        elif self.dataset_type == 'val' and args.val_data_pct > 0.0:
            self.data, self.labels = read_folder_of_authors(self.path,
                                                            self.args,
                                                            self.dataset_type)
        elif self.dataset_type == 'test' and args.test_data_pct > 0.0:
            self.data, self.labels = read_folder_of_authors(self.path,
                                                            self.args,
                                                            self.dataset_type)
        else:
            self.data, self.labels = None, None

        if args.model_name.startswith("BERT"):
            from transformers import AutoTokenizer, BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        else:
            self.tokenizer = None
    def __len__(self):
        return len(self.data)

    def __geitem__(self, idx):
        return self.data[idx], self.labels[idx]
