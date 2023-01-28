import pycocotools
import numpy as np


def rle_encoder(obj):
    rle_encode = pycocotools.mask.encode(obj)
    rle_encode['counts'] = rle_encode['counts'].decode()

    return rle_encode


def convert_to_coco(image_id, pred):
    instances = []

    boxes = pred['pred_boxes'].tensor.numpy()
    scores = pred['scores'].numpy()
    masks = pred['pred_masks'].numpy()

    for box, score, mask in zip(boxes, scores, masks):
        instances.append({
            'image_id': image_id,
            'box': box.tolist(),
            'score': float(score),
            'category_id': 1,
            'segmentation': rle_encoder(np.asfortranarray(mask))
        })

    return instances
