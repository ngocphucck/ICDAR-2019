import torch
from torch import nn
import json
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm


from .dataset import ReceiptDataset, collate_fn
from .model import CRNN


os.chdir('/home/doanphu/Documents/Code/Practice/FinalProject')


def train_batch(model, images, text_encodes, text_lens, optimizer, criterion, device):
    model.train()
    images = images.to(device)
    text_encodes = text_encodes.to(device)
    text_lens = text_lens.to(device)

    logits = model(images)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    batch_size = logits.size(1)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(text_lens)

    loss = criterion(log_probs, text_encodes, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    IMAGE_PATH = 'data/task2/image/images.json'
    TEXT_PATH = 'data/task2/annotation/texts.json'
    BATCH_SIZE = 16

    with open(IMAGE_PATH, 'r') as f:
        image_paths = json.load(f)
    with open(TEXT_PATH, 'r') as f:
        texts = json.load(f)

    X_train, X_test, y_train, y_test = train_test_split(image_paths, texts, test_size=0.3,
                                                        shuffle=True, random_state=2020)

    train_dataset = ReceiptDataset(X_train, y_train)
    val_dataset = ReceiptDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    n_epochs = 30
    lr = 1e-5
    model = CRNN()
    model.load_state_dict(torch.load('./crnn.pth', map_location=torch.device('cpu')))
    optimizer = Adam(model.parameters())
    criterion = nn.CTCLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_losses = list()
    for epoch in tqdm(range(n_epochs)):
        print(f'Epoch {epoch + 1}: ')
        train_batch_losses = list()
        for images, text_encodes, text_lens in train_dataloader:
            train_batch_losses.append(train_batch(model, images, text_encodes, text_lens, optimizer, criterion, device))

            print('Train batch loss: ', train_batch_losses[-1])

        train_losses.append(sum(train_batch_losses) / len(train_batch_losses))
        print('=================Train loss: {}======================'.format(train_losses[-1]))


if __name__ == '__main__':
    main()
    pass
