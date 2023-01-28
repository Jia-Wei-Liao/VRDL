import os
import time
import json
import torch
import argparse

from models.models import *
from utils.datasets import *


def get_device(cuda):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda}')

    else:
        device = torch.device('cpu')

    return device


def get_id(path):
    file_name = path.split('/')[-1]  # xxxx.png
    image_id = int(file_name[:-4])  # xxxx

    return image_id


def data_processing(image):
    image = torch.from_numpy(image).float().unsqueeze(0)
    image = image / 255  # normalize

    return image


def save_answer(save_list, save_path):
    with open(save_path, 'w') as file:
        json.dump(save_list, file, indent=4)

    return None


parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_path',
    type=str,
    default='../datasets/Testing/images',
    help='data_path'
    )

parser.add_argument(
    '-cfg',
    '--config',
    type=str,
    default='cfg/yolov4-pacsp.cfg',
    help='config'
    )

parser.add_argument(
    '--weight',
    type=str,
    default='runs/train/yolov4-pacsp/weights/best.pt',
    help='path of weight'
    )

parser.add_argument(
    '--device',
    type=int,
    default=0,
    help='cuda'
    )

parser.add_argument(
    '--image_size',
    type=int,
    default=640,
    help='image size'
    )


args = parser.parse_args()


if __name__ == '__main__':
    device = get_device(args.device)
    model = Darknet(args.config, args.image_size).to(device)
    model.load_state_dict(torch.load(args.weight)['model'])
    model.eval()
    test_set = LoadImages(
        args.data_path,
        img_size=args.image_size,
        auto_size=args.image_size)

    answer_list = []
    with torch.no_grad():
        start_time = time.time()
        for i, (path, image, origin_image, _) in enumerate(test_set):
            image_id = get_id(path)
            image = data_processing(image)
            pred = model(image.to(device), augment=True)[0]
            pred = non_max_suppression(pred)
            p = pred[0]
            p[:, :4] = scale_coords(
                image.shape[2:],
                p[:, :4],
                origin_image.shape)

            for *coords, score, category_id in p:
                x1, y1, x2, y2 = [float(e) for e in coords]
                w, h = x2-x1, y2-y1
                bbox = {
                    'image_id': image_id,
                    'bbox': [x1, y1, w, h],
                    'category_id': float(category_id),
                    'score': float(score)
                }
                answer_list.append(bbox)

            spend_time = time.time() - start_time
            print(f'iteration: {i+1}/{len(test_set)}, ',
                  f'time: {round(spend_time, 6)}')

    print(f'Inference time per image: {spend_time / len(test_set)}')
    save_answer(answer_list, 'answer.json')
