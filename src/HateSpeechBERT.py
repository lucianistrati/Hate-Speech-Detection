import torch.nn as nn
from transformers import AutoModel

class HateSpeechBERT(nn.Module):
    """ a class of fine tuned ro bert
    """

    def __init__(self, n_sentiments, seq_len=64, droput_prob=0.1):
        super().__init__()
        self.base_model = AutoModel.from_pretrained("bert-base-uncased")
        self.n_sentiments = n_sentiments

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(seq_len * 768, seq_len * 4)
        self.droput1 = nn.Dropout(p=droput_prob)
        self.linear2 = nn.Linear(seq_len * 4, n_sentiments)
        self.droput2 = nn.Dropout(p=droput_prob)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.base_model(input_ids=input_ids,
                                 attention_mask=attention_mask, \
                                 token_type_ids=token_type_ids)
        output = self.flatten(output.last_hidden_state)

        output = self.linear1(output)
        output = self.droput1(output)

        output = self.linear2(output)
        output = self.droput2(output)
        output = self.softmax(output)
        return output
