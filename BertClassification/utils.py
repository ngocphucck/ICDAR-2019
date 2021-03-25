import os
import json
from tqdm import tqdm


os.chdir('/home/doanphu/Documents/Code/Practice/FinalProject')


def create_data(data_path='data/raw_data/train', label_path='data/raw_data/train_task3',
                save_path='data/task3/train.json'):
    data = dict()
    classes = {'company': 0, 'date': 1, 'address': 2, 'total': 3, 'other': 4}
    found = 0
    not_found = 0

    for file in tqdm(os.listdir(data_path)):
        if file.endswith('txt') and file not in os.listdir(label_path):
            not_found += 1
        if file.endswith('txt') and file in os.listdir(label_path):
            found += 1
            with open(os.path.join(label_path, file), 'r') as f:
                labels = f.readlines()
                labels = ''.join(labels)
                labels = labels.split()
                labels = ''.join(labels)
                labels = labels.split('\"')
                labels = labels[1:-1:2]

            with open(os.path.join(data_path, file), 'r') as f:
                lines = f.readlines()

            for line in lines:
                line_split = line.split(',')
                coordinate = line_split[: 8]
                coordinate = ''.join(coordinate)
                text = line[len(coordinate) + 8:-1]
                join_text = ''.join(text.split())

                data[text] = 4
                for i in range(1, len(labels), 2):
                    if labels[i].find(join_text[: 5]) != -1 or join_text.find(labels[i]) != -1:
                        data[text] = classes[labels[i - 1]]
                        break

    with open(save_path, 'w') as f:
        json.dump(data, f)

    print(f'Found {found} files; not found {not_found} files')


def load_data(data_path='data/task3/train.json'):
    with open(data_path, 'r') as f:
        data = json.load(f)

    texts = list(data)
    labels = [data[text] for text in texts]

    return texts, labels


if __name__ == '__main__':
    with open('data/task3/train.json', 'r') as f:
        data = json.load(f)

    texts = [text for text in data.keys()]

    pass
