import os
import time
import argparse
import numpy as np
from PIL import Image

import torch

from src.transforms import test_transform
from src.model import FasterRCNN
from src.utils import *


parser = argparse.ArgumentParser()

parser.add_argument(
    '--ckpt_path',
    type=str,
    default='./checkpoint/example/epoch=0.pth',
    help='path of checkpoint')

parser.add_argument(
    '--save_root',
    type=str,
    default='',
    help='save root')

parser.add_argument(
    '--folder',
    type=str,
    default='./test',
    help='test image folder')

parser.add_argument(
    '--num_classes',
    type=int,
    default=11,
    help='number of classes')

parser.add_argument(
    '--cuda',
    type=int,
    default=1,
    help='cuda')

args = parser.parse_args()


if __name__ == '__main__':
    model = FasterRCNN(args.num_classes)
    model.load(args.ckpt_path)
    device = get_device(args.cuda)
    model.to(device)
    model.eval()

    test_files = os.listdir(args.folder)
    file_numbers = np.array([int(s[:-4]) for s in test_files])
    sort_idxs = np.argsort(file_numbers)
    test_files = np.array(test_files)[sort_idxs]
    predictions = []

    for i, file in enumerate(test_files):
        start_time = time.time()
        image = Image.open(os.path.join(args.folder, file))
        image = test_transform(image).unsqueeze(0).to(device)
        pred = model(image)
        boxes = pred[0]['boxes'].detach().cpu().numpy().tolist()
        labels = pred[0]['labels'].detach().cpu().numpy().tolist()
        scores = pred[0]['scores'].detach().cpu().numpy().tolist()

        for j in range(len(scores)):

            if labels[j] == 10:
                labels[j] = 0

            x1, y1, x2, y2 = bboxes[j]
            predictions.append({
                'image_id': int(file[:-4]),
                'bbox': [x1, y1, x2-x1, y2-y1],
                'score': scores[j],
                'category_id': labels[j],
                })

        spend_time = time.time() - start_time
        print(f'image: {i+1}/{test_files.size}, time: {spend_time:.6f}')

    save_path = os.path.join(
        '/'.join(args.ckpt_path.split('/')[:-1]), 'answer.json')
    save_json_file(predictions, save_path)
