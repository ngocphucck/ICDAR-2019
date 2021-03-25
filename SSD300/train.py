from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from dataset import TextDetectionDataset
from utils import get_images, get_objects
from model import SSD, MultiBoxLoss
from dataset import collate_fn


# 2.Dataloader
IMAGES_JSON_PATH = './data/train/images.json'
images = get_images(IMAGES_JSON_PATH)
OBJECTS_JSON_PATH = './data/train/objects.json'
objects = get_objects(OBJECTS_JSON_PATH)

BATCH_SIZE = 5

X_train, X_val, y_train, y_val = train_test_split(images, objects, test_size=0.3, random_state=2021)

train_dataset = TextDetectionDataset(X_train, y_train)
val_dataset = TextDetectionDataset(X_val, y_val)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
val_dataset = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# 3.Model
N_EPOCHS = 10
model = SSD()
loss_fn = MultiBoxLoss()
optimizer = Adam(model.parameters())

train_losses = []
for epoch in range(N_EPOCHS):
    for X_batch, y_batch in tqdm(train_dataloader):
        locs, cls_scores = model(X_batch)
        loss = loss_fn(locs, cls_scores, y_batch)
        train_losses.append(loss.item())
        print('\nTrain batch loss: ', loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
