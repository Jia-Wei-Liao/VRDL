import os
import glob
import tqdm
import numpy as np
from PIL import Image
from pycocotools import mask
from detectron2.structures import BoxMode
from src.coco_mask import rle_encoder
from src.utils import *


def get_mask_annot(image_mask):
    image_mask = np.array(Image.open(image_mask))
    binay_mask = np.asfortranarray(image_mask > 0)
    seg = rle_encoder(binay_mask)
    bbox_mode = BoxMode.XYWH_ABS
    bbox = mask.toBbox(seg).tolist()
    mask_annot = {
        'category_id': 0,
        'segmentation': seg,
        'bbox_mode': bbox_mode,
        'bbox': bbox
    }

    return mask_annot


def generate_train_annot(file_root):
    file_list = os.listdir(os.path.join(file_root, 'train'))
    image_annots = []

    for image_id, image_name in enumerate(tqdm.tqdm(file_list)):
        image_root = os.path.join(file_root, 'train', image_name)
        image_path = os.path.join(image_root, 'images', f'{image_name}.png')
        masks_path = glob.glob(os.path.join(image_root, 'masks', '*.png'))

        image = Image.open(image_path)
        image_height, image_width = image.size

        mask_annots = []
        for image_mask in masks_path:
            mask_annots.append(get_mask_annot(image_mask))

        annot = {
            'file_name': image_path,
            'height': image_height,
            'width': image_width,
            'image_id': image_id,
            'annotations': mask_annots
        }

        image_annots.append(annot)

    print('finish!')

    return image_annots


if __name__ == '__main__':
    file_root = 'dataset'
    image_annots = generate_train_annot(file_root)
    save_json(image_annots, os.path.join(file_root, 'train.json'))
