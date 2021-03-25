import torch
from PIL import Image, ImageOps
from scipy.special import logsumexp
import math
import numpy as np


from .utils import decode
from .dataset import ResizeNormalize
from .model import CRNN


NINF = -1 * float('inf')
DEFAULT_THRESHOLD = 0.01


def _reconstruct(labels, blank=0):
    new_labels = list()

    previous = None

    for char in labels:
        if char != previous:
            new_labels.append(char)
            previous = char

    new_labels = [char for char in new_labels if char != blank]

    return new_labels


def greedy_decode(log_probs, blank=0):
    labels = np.argmax(log_probs, axis=-1)
    labels = _reconstruct(labels, blank)

    labels = decode(labels)

    return labels


def beam_search_decode(log_probs, beam_size, threshold=math.log(DEFAULT_THRESHOLD), blank=0):
    length = log_probs.shape[0]
    n_classes = log_probs.shape[-1]

    beams = [([], 0)]

    for t in range(length):
        new_beams = list()
        for prefix, accumulate_log_prob in beams:
            for c in range(n_classes):
                log_prob = log_probs[t, c]
                if log_prob < threshold:
                    continue

                new_prefix = prefix + [c]
                new_accumulate_log_prob = accumulate_log_prob + log_prob
                new_beams.append((new_prefix, new_accumulate_log_prob))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[: beam_size]

    total_accu_log_prob = {}
    for prefix, accumulate_prob in beams:
        labels = tuple(_reconstruct(prefix))
        total_accu_log_prob[labels] = logsumexp([accumulate_prob, total_accu_log_prob.get(labels, NINF)])

    labels_beams = [(list(labels), accumulate_prob) for labels, accumulate_prob in total_accu_log_prob.items()]
    labels_beams.sort(key=lambda x: x[1], reverse=True)
    labels = labels_beams[0][0]

    labels = decode(labels)

    return labels


def predict(image, model):
    image = ImageOps.grayscale(image)

    normalize = ResizeNormalize()
    image = normalize(image)

    logits = model(image.unsqueeze(0))
    logits = logits.permute(0, 2, 1)
    shape = logits.size()
    logits = logits.view(shape[0], shape[1])
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs = log_probs.detach().numpy()

    pred_label = beam_search_decode(log_probs, beam_size=5)

    return pred_label


if __name__ == '__main__':
    IMAGE_PATH = 'data/task2/image/19288.jpg'
    image = Image.open(IMAGE_PATH)
    image.show()

    model = CRNN()
    model.load_state_dict(torch.load('./crnn.pth', map_location=torch.device('cpu')))

    pred_label = predict(IMAGE_PATH, model)
    print(pred_label)
    pass
