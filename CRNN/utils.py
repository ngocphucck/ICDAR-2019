import os
from PIL import Image
import json


os.chdir('/home/doanphu/Documents/Code/Practice/FinalProject')


def parse_annotation(annotation):
    boxes = list()
    texts = list()

    with open(annotation, 'r') as f:
        for line in f.readlines():
            line_spl = line.split(',')

            full_box = [int(line_spl[i]) for i in range(8)]
            box_len = ' '.join([str(element) for element in full_box])
            text = line[len(box_len) + 1:-1]

            box = [full_box[0], full_box[1], full_box[4], full_box[5]]

            boxes.append(box)
            texts.append(text)

    return boxes, texts


def create_data(data_directory='data/raw_data/train',
                image_directory='data/task2/image', annotation_directory='data/task2/annotation'):
    all_image_paths = list()
    all_texts = list()

    index = 0
    for file in os.listdir(data_directory):
        if file.endswith('.txt'):
            if file[: -4] + '.jpg' not in os.listdir(data_directory):
                continue

            annotation = os.path.join(os.getcwd(), data_directory, file)
            image_path = os.path.join(os.getcwd(), data_directory, file[: -4] + '.jpg')

            boxes, texts = parse_annotation(annotation)
            image = Image.open(image_path)

            for i, box in enumerate(boxes):
                index += 1
                crop_image_path = os.path.join(image_directory, str(index) + '.jpg')
                annotation_path = os.path.join(annotation_directory, str(index) + '.txt')
                all_image_paths.append(crop_image_path)

                crop_image = image.crop(box)
                crop_image.save(crop_image_path)

                with open(annotation_path, 'w') as f:
                    f.write(texts[i])
                    all_texts.append(texts[i])

    with open(os.path.join(image_directory, 'images.json'), 'w') as f:
        json.dump(all_image_paths, f)

    with open(os.path.join(annotation_directory, 'texts.json'), 'w') as f:
        json.dump(all_texts, f)


def create_vocab(annotation_directory='data/task2/annotation'):
    vocab = set()

    for file in os.listdir(annotation_directory):
        if not file.endswith('.txt'):
            continue
        with open(os.path.join(os.getcwd(), annotation_directory, file), 'r') as f:
            text = f.read()
            vocab.update(list(text))

    vocab = sorted(vocab)

    with open(os.path.join(os.getcwd(), annotation_directory, 'vocab.json'), 'w') as f:
        json.dump(list(vocab), f)

    return vocab


def create_map(vocab):
    map = {i + 1: char for i, char in enumerate(vocab)}
    rev_map = {char: i for i, char in map.items()}

    return map, rev_map


def encode(text):
    with open('data/task2/annotation/vocab.json', 'r') as f:
        vocab = json.load(f)

    map, rev_map = create_map(vocab)

    text_encode = [rev_map[text[i]] for i in range(len(text))]

    return text_encode, len(text)


def decode(labels):
    with open('data/task2/annotation/vocab.json', 'r') as f:
        vocab = json.load(f)

    map, rev_map = create_map(vocab)

    text_decode = [map[i] for i in labels]
    text_decode = ''.join(text_decode)

    return text_decode


if __name__ == '__main__':
    with open('data/task2/annotation/vocab.json', 'r') as f:
        vocab = json.load(f)

    print(create_map(vocab))
    pass
