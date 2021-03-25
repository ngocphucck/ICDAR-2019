import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, GaussianBlur, Resize, Normalize, ToTensor, Compose, ToPILImage
from PIL import Image

from utils import scale


class TextDetectionDataset(Dataset):
    def __init__(self, images, objects):
        super(TextDetectionDataset, self).__init__()
        self.images = images
        self.objects = objects

        self.transforms = Compose([
            ColorJitter(),
            GaussianBlur(kernel_size=5),
            Resize((300, 300)),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225))
        ])

    def __getitem__(self, item):
        image_path = self.images[item]
        boxes = self.objects[item]

        image = Image.open(image_path)
        shape = image.size
        image = self.transforms(image)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        boxes = scale(boxes, shape)

        return image, boxes

    def __len__(self):
        return len(self.images)


def collate_fn(batch):
    images = list()
    boxes = list()

    for b in batch:
        images.append(b[0])
        boxes.append(b[1])

    images = torch.stack(images, dim=0)

    return images, boxes


if __name__ == '__main__':
    dataset = TextDetectionDataset()
    print(dataset[0])
    pass
