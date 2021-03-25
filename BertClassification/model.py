import torch.nn as nn
from transformers import BertModel, BertForSequenceClassification


class TextClassification(nn.Module):
    def __init__(self, pretrained_name='bert-base-uncased', n_classes=5):
        super(TextClassification, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_name)
        self.dropout = nn.Dropout(0.1)
        self.classification = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, text_ids, text_attns):
        out = self.bert(text_ids, attention_mask=text_attns)
        out = out.last_hidden_state[:, 0, :]
        out = self.dropout(out)
        out = self.classification(out)

        return out
