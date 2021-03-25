import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import os
from tqdm import tqdm


from utils import load_data
from dataset import TextDataset
from model import TextClassification


os.chdir('/home/doanphu/Documents/Code/Practice/FinalProject')


# Data
BATCH_SIZE = 16
texts, labels = load_data()

train_dataset = TextDataset(texts, labels)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TextClassification()
lr = 1e-4
optimizer = Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()
n_epochs = 50

losses = []
for epoch in range(n_epochs):
    print(f'Epoch {epoch + 1}:')

    batch_losses = []
    for batch_text_ids, batch_text_attns, batch_labels in tqdm(train_dataloader):
        batch_text_ids = batch_text_ids.to(device)
        batch_text_attns = batch_text_attns.to(device)
        batch_labels = torch.flatten(batch_labels).to(device)

        pred = model(batch_text_ids, batch_text_attns)
        loss = loss_fn(pred, batch_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f'batch loss: {loss.item()}')
        batch_losses.append(loss.item())

    losses.append(sum(batch_losses) / len(batch_losses))
    print(f'Epoch loss: {losses[-1]}')
