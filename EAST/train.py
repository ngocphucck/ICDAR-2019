from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import json
import torch

from dataset import ReceiptDataset
from loss import Loss
from model import East

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Split train test
IMAGE_PATHS = 'EAST/data/images.json'
BOXES = 'EAST/data/boxes.json'

with open(IMAGE_PATHS, 'r') as f:
    image_paths = json.load(f)
with open(BOXES, 'r') as f:
    boxes = json.load(f)

BATCH_SIZE = 8

X_train, X_val, y_train, y_val = train_test_split(image_paths, boxes,
                                                  test_size=0.35, shuffle=True, random_state=2021)
train_dataset = ReceiptDataset(X_train, y_train)
val_dataset = ReceiptDataset(X_val, y_val)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
EPOCHS = 50
model = East().to(device)
model.load_state_dict(torch.load('east1.pt'))
lr = 1e-4
loss_fn = Loss().to(device)
optimizer = Adam(model.parameters(), lr=lr)
best_val_loss = 0.455

train_loss = list()
val_loss = list()

for epoch in range(EPOCHS):
    print('Epoch {}'.format(epoch + 1))

    train_batch_loss = list()
    for X_batch_train, gt_score, gt_geo in tqdm(train_dataloader):
        X_batch_train = X_batch_train.to(device)
        gt_score = gt_score.to(device)
        gt_geo = gt_geo.to(device)
        pred_score, pred_geo = model(X_batch_train)

        loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
        train_batch_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Train batch loss: ', train_batch_loss[-1])

    train_loss.append(sum(train_batch_loss) / len(train_batch_loss))
    print('***Train loss***: ', train_loss[-1])

    val_batch_loss = list()
    for X_batch_val, gt_score, gt_geo in val_dataloader:
        X_batch_val = X_batch_val.to(device)
        gt_score = gt_score.to(device)
        gt_geo = gt_geo.to(device)
        pred_score, pred_geo = model(X_batch_val)

        loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
        val_batch_loss.append(loss.item())

        torch.save(model, './east.pt')
        print('Val batch loss: ', val_batch_loss[-1])

    val_loss.append(sum(val_batch_loss) / len(val_batch_loss))
    print('***Validation loss**: ', val_loss[-1])

    if best_val_loss > val_loss[-1]:
      best_val_loss = val_loss[-1]
      torch.save(model.state_dict(), './east1.pt')
      print('Save!')

    print('best val loss: ', best_val_loss)
