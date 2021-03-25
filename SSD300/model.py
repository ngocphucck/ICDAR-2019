import torch
import torchvision
from torch import nn
from math import sqrt

from utils import drop, cxcy_to_gcxgcy, gcxgcy_to_cxcy, cxcy_to_xy, xy_to_cxcy
from dataset import TextDetectionDataset
from metrics import IoU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VGGBase(nn.Module):
    def __init__(self):
        super(VGGBase, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)  # (64, 300, 300)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)  # (64, 300, 300)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (64, 150, 150)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # (128, 150, 150)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  # (128, 150, 150)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (128, 75, 75)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)  # (256, 75, 75)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)  # (256, 75, 75)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)  # (256, 75, 75)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # (256, 38, 38)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)  # (512, 38, 38)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)  # (512, 38, 38)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)  # (512, 38, 38)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # (512, 19, 19)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)  # (512, 19, 19)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)  # (512, 19, 19)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)  # (512, 19, 19)
        self.pool5 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)  # (512, 19, 19)

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3,
                               padding=6, dilation=6)  # (1024, 19, 19)

        self.conv7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)  # (1024, 19, 19)

        self.relu = nn.ReLU()
        self.load_pretrained_layers()

    def forward(self, X):
        out = self.conv1_1(X)
        out = self.relu(out)
        out = self.conv1_2(out)
        out = self.relu(out)
        out = self.pool1(out)

        out = self.conv2_1(out)
        out = self.relu(out)
        out = self.conv2_2(out)
        out = self.relu(out)
        out = self.pool2(out)

        out = self.conv3_1(out)
        out = self.relu(out)
        out = self.conv3_2(out)
        out = self.relu(out)
        out = self.conv3_3(out)
        out = self.relu(out)
        out = self.pool3(out)

        out = self.conv4_1(out)
        out = self.relu(out)
        out = self.conv4_2(out)
        out = self.relu(out)
        out = self.conv4_3(out)
        out = self.relu(out)
        conv4_3_fm = out
        out = self.pool4(out)

        out = self.conv5_1(out)
        out = self.relu(out)
        out = self.conv5_2(out)
        out = self.relu(out)
        out = self.conv5_3(out)
        out = self.relu(out)
        out = self.pool5(out)

        out = self.conv6(out)
        out = self.relu(out)

        out = self.conv7(out)
        out = self.relu(out)
        conv7_fm = out

        return conv4_3_fm, conv7_fm

    def load_pretrained_layers(self):
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained vgg16
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Copy weights from pretrained model
        for i, param in enumerate(param_names[: -4]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        pretrained_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
        pretrained_fc6_bias = pretrained_state_dict['classifier.0.bias']
        state_dict['conv6.weight'] = drop(pretrained_fc6_weight, factors=[4, 1, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = drop(pretrained_fc6_bias, factors=[4])  # 1024

        pretrained_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)
        pretrained_fc7_bias = pretrained_state_dict['classifier.3.bias']
        state_dict['conv7.weight'] = drop(pretrained_fc7_weight, factors=[4, 4, 1, 1])
        state_dict['conv7.bias'] = drop(pretrained_fc7_bias, factors=[4])


class AuxiliaryConvolution(nn.Module):
    def __init__(self):
        super(AuxiliaryConvolution, self).__init__()
        self.conv8_1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)  # (256, 19, 19)
        self.conv8_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2)  # (512, 10, 10)

        self.conv9_1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1)  # (128, 10, 10)
        self.conv9_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2)  # (256, 5, 5)

        self.conv10_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)  # (128, 5, 5)
        self.conv10_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)  # (256, 3, 3)

        self.conv11_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)  # (128, 3, 3)
        self.conv11_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)  # (256, 1, 1)

        self.init_conv2d()

    def forward(self, X):
        out = self.conv8_1(X)
        out = self.conv8_2(out)
        conv8_2_fm = out

        out = self.conv9_1(out)
        out = self.conv9_2(out)
        conv9_2_fm = out

        out = self.conv10_1(out)
        out = self.conv10_2(out)
        conv10_2_fm = out

        out = self.conv11_1(out)
        out = self.conv11_2(out)
        conv11_2_fm = out

        return conv8_2_fm, conv9_2_fm, conv10_2_fm, conv11_2_fm

    def init_conv2d(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)


class PredictionConvolution(nn.Module):
    def __init__(self):
        super(PredictionConvolution, self).__init__()

        # Number of priors in each feature map
        n_priors = {
            'conv4_3': 4,
            'conv7': 6,
            'conv8_2': 6,
            'conv9_2': 6,
            'conv10_2': 4,
            'conv11_2': 4
        }

        # Localization layer
        self.loc_conv4_3 = nn.Conv2d(in_channels=512, out_channels=4 * n_priors['conv4_3'],
                                     kernel_size=3, padding=1)  # (16, 38, 38)
        self.loc_conv7 = nn.Conv2d(in_channels=1024, out_channels=4 * n_priors['conv7'],
                                   kernel_size=3, padding=1)  # (24, 19, 19)
        self.loc_conv8_2 = nn.Conv2d(in_channels=512, out_channels=4 * n_priors['conv8_2'],
                                     kernel_size=3, padding=1)  # (24, 10, 10)
        self.loc_conv9_2 = nn.Conv2d(in_channels=256, out_channels=4 * n_priors['conv9_2'],
                                     kernel_size=3, padding=1)  # (24, 5, 5)
        self.loc_conv10_2 = nn.Conv2d(in_channels=256, out_channels=4 * n_priors['conv10_2'],
                                      kernel_size=3, padding=1)  # (16, 3, 3)
        self.loc_conv11_2 = nn.Conv2d(in_channels=256, out_channels=4 * n_priors['conv11_2'],
                                      kernel_size=3, padding=1)  # (16, 1, 1)

        # Classification layer
        self.cls_conv4_3 = nn.Conv2d(in_channels=512, out_channels=2 * n_priors['conv4_3'],
                                     kernel_size=3, padding=1)  # (4, 38, 38)
        self.cls_conv7 = nn.Conv2d(in_channels=1024, out_channels=2 * n_priors['conv7'],
                                   kernel_size=3, padding=1)  # (6, 19, 19)
        self.cls_conv8_2 = nn.Conv2d(in_channels=512, out_channels=2 * n_priors['conv8_2'],
                                     kernel_size=3, padding=1)  # (6, 10, 10)
        self.cls_conv9_2 = nn.Conv2d(in_channels=256, out_channels=2 * n_priors['conv9_2'],
                                     kernel_size=3, padding=1)  # (6, 5, 5)
        self.cls_conv10_2 = nn.Conv2d(in_channels=256, out_channels=2 * n_priors['conv10_2'],
                                      kernel_size=3, padding=1)  # (4, 3, 3)
        self.cls_conv11_2 = nn.Conv2d(in_channels=256, out_channels=2 * n_priors['conv11_2'],
                                      kernel_size=3, padding=1)  # (4, 1, 1)

        self.init_conv2d()

    def forward(self, conv4_3_fm, conv7_fm, conv8_2_fm, conv9_2_fm, conv10_2_fm, conv11_2_fm):

        # Localization prediction
        l_conv4_3 = self.loc_conv4_3(conv4_3_fm)  # (N, 4 * 4, 38, 38)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()
        l_conv4_3 = l_conv4_3.view(conv4_3_fm.size(0), -1, 4)

        l_conv7 = self.loc_conv7(conv7_fm)  # (N, 6 * 4, 19, 19)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()
        l_conv7 = l_conv7.view(conv7_fm.size(0), -1, 4)

        l_conv8_2 = self.loc_conv8_2(conv8_2_fm)  # (N, 6 * 4, 10, 10)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()
        l_conv8_2 = l_conv8_2.view(conv8_2_fm.size(0), -1, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_fm)  # (N, 6 * 4, 5, 5)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()
        l_conv9_2 = l_conv9_2.view(conv9_2_fm.size(0), -1, 4)

        l_conv10_2 = self.loc_conv10_2(conv10_2_fm)  # (N, 4 * 4, 3, 3)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()
        l_conv10_2 = l_conv10_2.view(conv10_2_fm.size(0), -1, 4)

        l_conv11_2 = self.loc_conv11_2(conv11_2_fm)  # (N, 4 * 4, 1, 1)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()
        l_conv11_2 = l_conv11_2.view(conv11_2_fm.size(0), -1, 4)

        # Classification prediction
        c_conv4_3 = self.cls_conv4_3(conv4_3_fm)  # (N, 4 * 2, 38, 38)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()
        c_conv4_3 = c_conv4_3.view(conv4_3_fm.size(0), -1, 2)

        c_conv7 = self.cls_conv7(conv7_fm)  # (N, 6 * 2, 19, 19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()
        c_conv7 = c_conv7.view(conv7_fm.size(0), -1, 2)

        c_conv8_2 = self.cls_conv8_2(conv8_2_fm)  # (N, 6 * 2, 10, 10)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()
        c_conv8_2 = c_conv8_2.view(conv8_2_fm.size(0), -1, 2)

        c_conv9_2 = self.cls_conv9_2(conv9_2_fm)  # (N, 6 * 2, 5, 5)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()
        c_conv9_2 = c_conv9_2.view(conv9_2_fm.size(0), -1, 2)

        c_conv10_2 = self.cls_conv10_2(conv10_2_fm)  # (N, 4 * 2, 3, 3)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()
        c_conv10_2 = c_conv10_2.view(conv10_2_fm.size(0), -1, 2)

        c_conv11_2 = self.cls_conv11_2(conv11_2_fm)  # (N, 4 * 2, 1, 1)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()
        c_conv11_2 = c_conv11_2.view(conv11_2_fm.size(0), -1, 2)

        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)
        cls_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2], dim=1)

        return locs, cls_scores

    def init_conv2d(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)


class SSD(nn.Module):
    def __init__(self):
        super(SSD, self).__init__()
        self.vgg_base = VGGBase()
        self.auxiliary_convolution = AuxiliaryConvolution()
        self.prediction_convolution = PredictionConvolution()

        self.priors_cxcy = create_priors()

    def forward(self, X):
        conv4_3_fm, conv7_fm = self.vgg_base(X)
        conv8_2_fm, conv9_2_fm, conv10_2_fm, conv11_2_fm = self.auxiliary_convolution(conv7_fm)
        locs, cls_scores = self.prediction_convolution(conv4_3_fm, conv7_fm, conv8_2_fm,
                                                       conv9_2_fm, conv10_2_fm, conv11_2_fm)

        return locs, cls_scores

    def detect(self, predicted_locs, predicted_scores, threshold, max_overlap):
        batch_size = predicted_locs.size(0)
        predicted_scores = torch.nn.functional.softmax(predicted_scores, dim=2)  # (batch_size, 8732, 2)

        all_image_boxes = list()
        all_image_scores = list()

        for i in range(batch_size):
            decode_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy)
            )

            image_boxes = list()
            image_scores = list()

            text_scores = predicted_scores[i][:, 1]
            score_above_threshold = text_scores > threshold
            n_score_above_threshold = score_above_threshold.sum().item()

            text_scores = text_scores[score_above_threshold]
            text_decoded_locs = decode_locs[score_above_threshold]

            overlap = IoU(xy_to_cxcy(text_decoded_locs), xy_to_cxcy(text_decoded_locs))

            suppress = torch.zeros(n_score_above_threshold, dtype=torch.uint8).to(device)

            for box in range(text_decoded_locs.size(0)):
                if suppress[box] == 1:
                    continue

                suppress, _ = torch.max(suppress, overlap[box] > max_overlap)
                suppress[box] = 0

            image_boxes.append(text_decoded_locs[1 - suppress])
            image_scores.append(text_scores[1 - suppress])

            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([0., 0., 1., 1.]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)

            all_image_boxes.append(image_boxes)
            all_image_scores.append(image_scores)

        return all_image_boxes, all_image_scores


class MultiBoxLoss(nn.Module):
    def __init__(self, threshold=0.5, neg_pos_ratio=3, alpha=1):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = create_priors()
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (batch_size, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (batch_size, 8732)

        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = IoU(boxes[i], self.priors_cxcy)  # (n_objects, 8732)

            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)
            _, prior_for_each_object = overlap.max(dim=1)  # (n_objects)

            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)  # (8732)
            overlap_for_each_prior[prior_for_each_object] = 1  # (8732)

            label_for_each_prior = torch.zeros(n_priors)  # (8732)
            label_for_each_prior[object_for_each_prior] = 1
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0

            true_classes[i] = label_for_each_prior
            true_locs[i] = cxcy_to_gcxgcy(boxes[i][object_for_each_prior], self.priors_cxcy)

        positive_priors = true_classes != 0  # (batch_size, 8732)
        print(predicted_locs.size())
        # localization loss
        loc_loss = self.smooth_l1(predicted_locs[positive_priors],
                                  true_locs[positive_priors])

        n_positives = positive_priors.sum(dim=1)  # (batch_size)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (batch_size)

        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, 2), true_classes.view(-1))  # (batch_size * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (batch_size, 8732)

        conf_loss_pos = conf_loss_all[positive_priors]

        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (batch_size, 8732)

        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).\
            expand_as(conf_loss_neg).to(device)  # (batch_size, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (batch_size, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]

        conf_loss = (conf_loss_pos.sum() + conf_loss_hard_neg.sum()) / n_positives.sum().float()

        return conf_loss + self.alpha * loc_loss


def create_priors():
    fm_dims = {
        'conv4_3': 38,
        'conv7': 19,
        'conv8_2': 10,
        'conv9_2': 5,
        'conv10_2': 3,
        'conv11_2': 1
    }

    prior_scales = {
        'conv4_3': 0.1,
        'conv7': 0.2,
        'conv8_2': 0.375,
        'conv9_2': 0.55,
        'conv10_2': 0.725,
        'conv11_2': 0.9
    }

    aspect_ratios = {
        'conv4_3': [1.0, 2.0/1, 1.0/2],
        'conv7': [1.0, 2.0/1, 1.0/2, 3.0/1, 1.0/3],
        'conv8_2': [1.0, 2.0/1, 1.0/2, 3.0/1, 1.0/3],
        'conv9_2': [1.0, 2.0/1, 1.0/2, 3.0/1, 1.0/3],
        'conv10_2': [1.0, 2.0/1, 1.0/2],
        'conv11_2': [1.0, 2.0/1, 1.0/2]
    }

    priors = list()
    fm_names = list(fm_dims.keys())

    for k, fm_name in enumerate(fm_names):
        for i in range(fm_dims[fm_name]):
            for j in range(fm_dims[fm_name]):
                cx = (i + 0.5) / fm_dims[fm_name]
                cy = (j + 0.5) / fm_dims[fm_name]

                for ratio in aspect_ratios[fm_name]:
                    priors.append([cx, cy, prior_scales[fm_name] * sqrt(ratio),
                                   prior_scales[fm_name] / sqrt(ratio)])
                    if ratio == 1.0:
                        try:
                            additional_scale = sqrt(prior_scales[fm_names[k]] * prior_scales[fm_names[k + 1]])
                        except IndexError:
                            additional_scale = 1
                        priors.append([cx, cy, additional_scale * sqrt(ratio),
                                       additional_scale / sqrt(ratio)])

    priors = torch.FloatTensor(priors).to(device)
    priors.clamp_(0, 1)
    print('priors size: ', priors.size())
    return priors


if __name__ == '__main__':
    model = SSD()
    dataset = TextDetectionDataset()
    print(model(dataset[0][0].view(1, 3, 300, 300)))
    pass
