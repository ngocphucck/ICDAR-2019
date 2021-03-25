import json
import os
import torch
from metrics import IoU


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label_map = {'background': 0, 'text': 1}
rev_label_map = {0: 'background', 1: 'text'}
label_color_map = {'background': '#f58231', 'text': '#808080'}


def parse_annotation(annotation_path):
    boxes = list()

    with open(annotation_path, 'r') as f:
        for line in f.readlines():
            dat = line.split(',')

            xmin = int(dat[0])
            ymin = int(dat[1])
            xmax = int(dat[2])
            ymax = int(dat[5])

            boxes.append([xmin, ymin, xmax, ymax])

    return boxes


def create_data_lists(raw_data_directory='data/raw_data/train',
                      image_paths_json='SSD300/data/train/images.json',
                      objects_json='SSD300/data/train/objects.json'):
    images = list()
    objects = list()

    for file_name in os.listdir(raw_data_directory):
        if file_name.endswith('.txt'):
            file_name = file_name[:-4]

            if file_name + '.jpg' not in os.listdir(raw_data_directory):
                continue
            image_path = os.path.join(os.getcwd(), raw_data_directory, file_name + '.jpg')
            annotation_path = os.path.join(raw_data_directory, file_name + '.txt')
            annotation = parse_annotation(annotation_path)

            images.append(image_path)
            objects.append(annotation)

    with open(image_paths_json, 'w') as f:
        json.dump(images, f)

    with open(objects_json, 'w') as f:
        json.dump(objects, f)


def get_images(images_path):
    with open(images_path, 'r') as f:
         images = json.load(f)

    return images


def get_objects(objects_path):
    with open(objects_path, 'r') as f:
        objects = json.load(f)

    return objects


def scale(boxes, size):
    boxes[:, 0] = boxes[:, 0] / size[0]
    boxes[:, 1] = boxes[:, 1] / size[1]
    boxes[:, 2] = boxes[:, 2] / size[0]
    boxes[:, 3] = boxes[:, 3] / size[1]

    return boxes


def xy_to_cxcy(xy):
    cx = (xy[:, 0] + xy[:, 2]) / 2
    cy = (xy[:, 1] + xy[:, 3]) / 2
    w = xy[:, 2] - xy[:, 0]
    h = xy[:, 3] - xy[:, 1]

    return torch.stack([cx, cy, w, h], dim=1)


def cxcy_to_xy(cxcy):
    xmin = cxcy[:, 0] - cxcy[:, 2] / 2
    xmax = cxcy[:, 0] + cxcy[:, 2] / 2
    ymin = cxcy[:, 1] - cxcy[:, 3] / 2
    ymax = cxcy[:, 1] + cxcy[:, 3] / 2

    return torch.stack([xmin, ymin, xmax, ymax], dim=1)


def cxcy_to_gcxgcy(boxes, priors):
    gcx = (boxes[:, 0] - priors[:, 0]) / priors[:, 2]
    gcy = (boxes[:, 1] - priors[:, 1]) / priors[:, 3]
    gw = torch.log(boxes[:, 2] / priors[:, 2])
    gh = torch.log(boxes[:, 3] / priors[:, 3])

    return torch.stack([gcx, gcy, gw, gh], dim=1)


def gcxgcy_to_cxcy(g, priors):
    cx = g[:, 0] * priors[:, 2] + priors[:, 0]
    cy = g[:, 1] * priors[:, 3] + priors[:, 1]
    w = torch.exp(g[:, 2]) * priors[:, 2]
    h = torch.exp(g[:, 3]) * priors[:, 3]

    return torch.stack([cx, cy, w, h], dim=1)


def drop(tensor, factors):
    for dim, factor in enumerate(factors):
        tensor = torch.index_select(tensor, dim=dim, index=torch.tensor(range(0, tensor.size()[dim], factor)))

    return tensor


def calculate_mAP(det_boxes, det_scores, true_boxes):

    true_images = list()
    for i in range(len(true_boxes)):
        true_images.extend([i] * true_boxes[i].size(0))  # Correspond image
    true_images = torch.LongTensor(true_images).to(device)
    true_boxes = torch.cat(true_boxes, dim=0)

    det_images = list()
    for i in range(len(det_boxes)):
        det_images.extend([i] * det_boxes[i].size(0))  # Correspond image
    det_images = torch.LongTensor(det_images).to(device)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_objects, 4)
    det_scores = torch.cat(det_scores, dim=0)  # (n_objects, 1)
    n_objects = det_boxes.size(0)

    # Calculate APs
    det_scores, sort_ind = det_scores.sort(det_scores, descending=True)  # sorted with order of descending scores
    det_images = det_images[sort_ind]
    det_boxes = det_boxes[sort_ind]

    true_positives = torch.zeros(n_objects, dtype=torch.float).to(device)

    for id in range(n_objects):
        box = det_boxes[id].unsqueeze(0)
        image_id = det_images[id]

        object_boxes = true_boxes[true_images == image_id]
        if object_boxes.size(0) == 0:
            continue

        overlaps = IoU(xy_to_cxcy(box), xy_to_cxcy(object_boxes))
        max_overlap, _ = torch.max(overlaps.unsqueeze(0), dim=0)

        if max_overlap.item() > 0.5:
            true_positives[id] = 1

    # Compute cumulative precision and recall at each detection in order of decreasing scores
    cumul_true_positives = torch.cumsum(true_positives, dim=0)
    cumul_precision = cumul_true_positives / true_positives.size(0)
    cumul_recall = cumul_true_positives / true_positives.sum().item()

    recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()
    precisions = torch.zeros(len(recall_thresholds), dtype=torch.float).to(device)

    for i, threshold in enumerate(recall_thresholds):
        recall_above_threshold = cumul_recall >= threshold
        if recall_above_threshold.any():
            precisions[i] = cumul_precision[recall_above_threshold].max().item()
        else:
            precisions[i] = 0

    average_precision = precisions.mean()

    return average_precision


if __name__ == '__main__':
    create_data_lists()
    pass
