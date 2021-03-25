import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize
from PIL import Image, ImageOps
import numpy as np
import os


from .utils import encode


os.chdir('/home/doanphu/Documents/Code/Practice/FinalProject')


class ReceiptDataset(Dataset):
    def __init__(self, image_paths, texts, width=280, height=64):
        self.image_paths = image_paths
        self.texts = texts
        self.transform = ResizeNormalize(width, height)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = Image.open(image_path)
        image = ImageOps.grayscale(image)
        text = self.texts[item]

        image = self.transform(image)

        return image, text

    def __len__(self):
        return len(self.image_paths)


class ResizeNormalize(object):
    def __init__(self, width=280, height=64):
        self.scale_width = width
        self.scale_height = height
        self.transforms = transforms.Compose([
            ToTensor(),
            Normalize(mean=0.5,
                      std=0.5)
        ])

    def __call__(self, image):
        w, h = image.size
        new_height = self.scale_height
        new_width = w * (new_height / h)
        new_width = int(new_width)

        if new_width >= self.scale_width:
            image = image.resize((self.scale_width, self.scale_height))
        else:
            image = image.resize((new_width, new_height))
            image_pad = np.zeros((self.scale_height, self.scale_width))
            image_pad[: new_height, : new_width] = image
            image = image_pad
            image = Image.fromarray(np.uint8(image))

        image = self.transforms(image)
        return image


def collate_fn(batch):
    images = list()
    text_encodes = list()
    text_lens = list()
    for b in batch:
        images.append(b[0])
        text_encode, text_len = encode(b[1])
        text_encodes += text_encode
        text_lens.append(text_len)

    return torch.stack(images, dim=0), torch.tensor(text_encodes), torch.tensor(text_lens)


if __name__ == '__main__':
    dataset = ReceiptDataset(['data/task2/image/37553.jpg'], ['abc'])
    print(dataset[0][0].size())
    pass
