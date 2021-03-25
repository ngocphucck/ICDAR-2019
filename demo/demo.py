import os
import torch
from PIL import Image, ImageDraw


from EAST.model import East
from EAST.predict import detect, plot_boxes
from CRNN.model import CRNN
from CRNN.predict import predict
from BertClassification.model import TextClassification
from BertClassification.predict import category_predict


os.chdir('/home/doanphu/Documents/Code/Practice/FinalProject')


def demo(image_path, east_model_path='east1.pt', crnn_model_path='crnn.pth', bert_model_path='bertCls.pth'):
    orig_image = Image.open(image_path)
    image = Image.open(image_path)

    text_detector_model = East()
    text_detector_model.load_state_dict(torch.load(east_model_path, map_location=torch.device('cpu')))
    boxes = detect(image, text_detector_model, device=torch.device('cpu'))
    plot_boxes(image, boxes)

    ocr_model = CRNN()
    ocr_model.load_state_dict(torch.load(crnn_model_path, map_location=torch.device('cpu')))

    bert_model = TextClassification()
    bert_model.load_state_dict(torch.load(bert_model_path, map_location=torch.device('cpu')))

    for box in boxes:
        crop_image = orig_image.crop((box[0], box[1], box[4], box[5]))
        pred_text = predict(crop_image, ocr_model)

        category = category_predict(bert_model, pred_text)
        print(category)

        draw = ImageDraw.Draw(image)
        draw.text((box[0] - 10, box[1] - 10), pred_text, (255, 0, 0))

        if category[1] == 'other':
            continue

        draw.text((box[2] -20, box[3] - 10), category[1], (0, 0, 255))

    image.save('demo.jpg')
    pass


if __name__ == '__main__':
    IMAGE_PATH = 'data/raw_data/test/X51005337867.jpg'
    demo(IMAGE_PATH)
    pass
