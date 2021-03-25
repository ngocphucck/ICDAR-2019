from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
import json
import os
import torch

from .utils import rotate_img, get_score_geo, resize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReceiptDataset(Dataset):
    def __init__(self, image_paths, boxes, scale=0.25, length=512):
        super(ReceiptDataset, self).__init__()
        self.image_paths = image_paths
        self.boxes = boxes
        self.scale = scale
        self.length = length
        self.tranforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        vertices = np.array(self.boxes[item], dtype=int)

        image = Image.open(self.image_paths[item])
        image = image.convert('RGB')
        # image, vertices = rotate_img(image, vertices)
        image, vertices = resize(image, vertices, self.length)

        score_map, geo_map = get_score_geo(image, vertices, self.scale, self.length)
        image = self.tranforms(image)

        return image, score_map, geo_map

    def __len__(self):
        return len(self.image_paths)


if __name__ == '__main__':
    with open('EAST/data/images.json', 'r') as f:
        image_paths = json.load(f)
    with open('EAST/data/boxes.json', 'r') as f:
        boxes = json.load(f)

    dataset = ReceiptDataset(image_paths=[image_paths[0]], boxes=[boxes[0]])
    print(dataset[0])
