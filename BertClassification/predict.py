import torch
import os
from transformers import BertTokenizer


os.chdir('/home/doanphu/Documents/Code/Practice/FinalProject')
class_to_index = {'company': 0, 'date': 1, 'address': 2, 'total': 3, 'other': 4}
index_to_class = {0: 'company', 1: 'date', 2: 'address', 3: 'total', 4: 'other'}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def category_predict(model, text):
    text_tokenizer = tokenizer.encode_plus(text, add_special_tokens=True, padding=True, pad_to_multiple_of=32)
    text_id = text_tokenizer['input_ids']
    text_id = torch.tensor(text_id)
    text_id = text_id.unsqueeze(0)

    text_attn = text_tokenizer['attention_mask']
    text_attn = torch.tensor(text_attn)
    text_attn = text_id.unsqueeze(0)

    pred = model(text_id, text_attn)
    label_pred = torch.argmax(pred)

    return text, index_to_class[label_pred.item()]
