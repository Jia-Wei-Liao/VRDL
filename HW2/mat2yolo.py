import os
import argparse
import scipy.io as sio
from PIL import Image


def get_yolo_bbox(bboxes, image_width, image_height):
    yolo_bbox = []
    for bbox in bboxes:
        bbox = [e.squeeze().tolist() for e in bbox]
        h, l, t, w, label = bbox

        if label == 10:
            label = 0

        xc = (l+w/2) / image_width
        yc = (t+h/2) / image_height
        width = w / image_width
        height = h / image_height
        bbox = f'{label} {xc} {yc} {width} {height}'
        yolo_bbox.append(bbox)
        # print(bbox)

    yolo_bboxes = '\n'.join(yolo_bbox)

    return yolo_bboxes


def save_txt(save_content, save_path):
    with open(save_path, 'w') as f:
        f.write(save_content)

    return None


parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir',
    type=str,
    default='train',
    help='data_location'
    )

parser.add_argument(
    '--digit_struct',
    type=str,
    default='new_digitStruct.mat',
    help='digit structure'
    )

args = parser.parse_args()


if __name__ == '__main__':
    digit_struct = sio.loadmat(args.digit_struct)['digitStruct'][0]
    for i, b in enumerate(digit_struct):
        name, bboxes = b[0][0], b[1][0]
        image = Image.open(os.path.join(args.data_dir, name))
        image_width, image_height = image.size
        yolo_bbox = get_yolo_bbox(bboxes, image_width, image_height)
        save_path = os.path.join(args.data_dir, name.replace('png', 'txt'))
        save_txt(yolo_bbox, save_path)
        print(f'[{i+1}/{len(digit_struct)}]')
