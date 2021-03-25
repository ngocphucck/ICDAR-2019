import torch


def IoU(boxes1, boxes2):

    overlap = torch.zeros((boxes1.size(0), boxes2.size(0)))
    boxes1 = boxes1.expand(boxes2.size(0), boxes1.size(0), 4)
    boxes1 = boxes1.permute(1, 0, 2)

    boxes2 = boxes2.expand(boxes1.size(0), boxes2.size(0), 4)

    inter_width = boxes1[:, :, 2] / 2 + boxes2[:, :, 2] / 2 - torch.abs(boxes1[:, :, 0] - boxes2[:, :, 0])
    inter_width[inter_width < 0] = 0

    inter_height = boxes1[:, :, 3] / 2 + boxes2[:, :, 3] / 2 - torch.abs(boxes1[:, :, 1] - boxes2[:, :, 1])
    inter_height[inter_height < 0] = 0

    inter_area = inter_width * inter_height
    boxes1_area = boxes1[:, :, 2] * boxes1[:, :, 3]
    boxes2_area = boxes2[:, :, 2] * boxes2[:, :, 3]
    union_area = boxes1_area + boxes2_area - inter_area

    overlap = inter_area / union_area

    return overlap
