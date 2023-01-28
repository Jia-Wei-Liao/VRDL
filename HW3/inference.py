import os
import tqdm
import argparse
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import detection_utils
from src.coco_mask import convert_to_coco
from src.utils import *


def inference_step(model, test_img_ids):
    answer_list = []
    for test_image in tqdm.tqdm(test_img_ids):
        image_path = os.path.join('dataset', 'test', test_image['file_name'])
        image = detection_utils.read_image(image_path, format='BGR')
        test_image_id = test_image['id']
        pred = model(image)['instances'].to('cpu').get_fields()
        instances = convert_to_coco(test_image_id, pred)
        answer_list += instances

    return answer_list


def main(args):
    cfg = get_cfg()
    cfg.merge_from_file(f'checkpoint/{args.checkpoint}/config.yaml')
    cfg.MODEL.WEIGHTS = os.path.join(
        'checkpoint',
        args.checkpoint,
        args.weight_name)

    model = DefaultPredictor(cfg)
    test_img_ids = read_json(args.dataset)
    answer_list = inference_step(model, test_img_ids)
    save_root = os.path.join(
        'checkpoint', args.checkpoint,
        f'submission_{args.weight_name[:-4]}')
    save_path = os.path.join(save_root, 'answer.json')
    os.makedirs(save_root, exist_ok=True)
    save_json(answer_list, save_path)
    print('finish!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='2021-12-10-09-43',
        help='path of checkpoint')

    parser.add_argument(
        '--weight_name',
        type=str,
        default='model_final.pth',
        help='name of weight')

    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset/test_img_ids.json',
        help='path of dataset')

    args = parser.parse_args()
    main(args)
