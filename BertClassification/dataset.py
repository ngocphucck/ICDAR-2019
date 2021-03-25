import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class TextDataset(Dataset):
    def __init__(self, texts, classes, max_len=32):
        self.texts = texts
        self.classes = classes
        self.max_len = 32

    def __getitem__(self, item):
        text = self.texts[item]
        text = tokenizer.tokenize(text)
        print(text)
        text = tokenizer.encode_plus(text, add_special_tokens=True, padding=True, pad_to_multiple_of=self.max_len)
        text_ids = text['input_ids']
        text_attn = text['attention_mask']

        text_ids = torch.tensor(text_ids, dtype=torch.long)
        text_attn = torch.tensor(text_attn, dtype=torch.long)
        label = torch.tensor(self.classes[item])

        return text_ids, text_attn, label

    def __len__(self):
        return len(self.classes)


if __name__ == '__main__':
    dataset = TextDataset(['Myajgdiahghhe'], [3])
    print(dataset[0])
    pass
