import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import os
import numpy as np
import lanms


from .model import East
from .utils import get_rotate_mat

os.chdir('/home/doanphu/Documents/Code/Practice/FinalProject')


def resize_img(img):
	w, h = img.size
	resize_w = w
	resize_h = h

	resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
	resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
	img = img.resize((resize_w, resize_h), Image.BILINEAR)
	ratio_h = resize_h / h
	ratio_w = resize_w / w

	return img, ratio_h, ratio_w


def load_pil(img):
	t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
	return t(img).unsqueeze(0)


def is_valid_poly(res, score_shape, scale):
	cnt = 0
	for i in range(res.shape[1]):
		if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or \
				res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
			cnt += 1
	return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
	polys = []
	index = []
	valid_pos *= scale
	d = valid_geo[:4, :]  # 4 x N

	for i in range(valid_pos.shape[0]):
		x = valid_pos[i, 0]
		y = valid_pos[i, 1]
		y_min = y - d[0, i] * 1.3
		y_max = y + d[1, i] * 1.3
		x_min = x - d[2, i] * 1.1
		x_max = x + d[3, i] * 1.1

		temp_x = np.array([[x_min, x_max, x_max, x_min]])
		temp_y = np.array([[y_min, y_min, y_max, y_max]])

		coordinate = np.concatenate((temp_x, temp_y), axis=0)

		if is_valid_poly(coordinate, score_shape, scale):
			index.append(i)
			polys.append([coordinate[0, 0], coordinate[1, 0], coordinate[0, 1], coordinate[1, 1],
						coordinate[0, 2], coordinate[1, 2], coordinate[0, 3], coordinate[1, 3]])

	return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
	score = score[0, :, :]
	xy_text = np.argwhere(score > score_thresh)
	if xy_text.size == 0:
		return None

	xy_text = xy_text[np.argsort(xy_text[:, 0])]
	valid_pos = xy_text[:, ::-1].copy()
	valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]
	polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape) 
	if polys_restored.size == 0:
		return None

	boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
	boxes[:, :8] = polys_restored
	boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
	boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)

	return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
	if boxes is None or boxes.size == 0:
		return None
	boxes[:, [0, 2, 4, 6]] /= ratio_w
	boxes[:, [1, 3, 5, 7]] /= ratio_h
	return np.around(boxes)
	
	
def detect(img, model, device):
	img, ratio_h, ratio_w = resize_img(img)
	with torch.no_grad():
		score, geo = model(load_pil(img).to(device))
	boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
	return adjust_ratio(boxes, ratio_w, ratio_h)


def plot_boxes(img, boxes):
	if boxes is None:
		return img
	
	draw = ImageDraw.Draw(img)
	for box in boxes:
		draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0, 255, 0))
	return img


if __name__ == '__main__':
	model_path = './east1.pt'
	img_path = 'data/raw_data/test/X51005230616.jpg'
	res_img = './res.png'

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = East().to(device)
	model.load_state_dict(torch.load('./east1.pt', map_location=torch.device('cpu')))
	model.eval()
	img = Image.open(img_path)

	boxes = detect(img, model, device)

	plot_img = plot_boxes(img, boxes)
	plot_img.save(res_img)
