import os
import yaml
import argparse
from detectron2.data import MetadataCatalog, DatasetCatalog
from src.trainer import Trainer
from src.utils import *


def main(args):
    cfg = set_cfg(args)
    DatasetCatalog.register("Nuclei_data", lambda: read_json(args.dataset))
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset/train.json',
        help='path of dataset')

    parser.add_argument(
        '-tn',
        '--train_num',
        type=int,
        default=24,
        help='number of training data')

    parser.add_argument(
        '--model',
        type=str,
        default='mask_rcnn_R_50_C4_1x',
        help='mask_rcnn_R_X_X')

    parser.add_argument(
        '--epoch',
        type=int,
        default=120,
        help='number of epoch')

    parser.add_argument(
        '-bs',
        '--batch_size',
        type=int,
        default=1,
        help='number of batch size')

    parser.add_argument(
        '-lr',
        '--base_lr',
        type=float,
        default=0.01,
        help='base learning rate')

    parser.add_argument(
        '-df',
        '--decay_factor',
        type=float,
        default=0.1,
        help='learning rate decay factor')

    parser.add_argument(
        '--warmup_ep',
        type=int,
        default=3,
        help='epoch of warm up learning rate')

    parser.add_argument(
        '-ld_ep',
        '--lr_decay_ep',
        type=list,
        default=(20, 50, 80, 90),
        help='epoch of learning rate decay')

    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='cpu or cuda: 0, 1, 2, 3')

    parser.add_argument(
        '--save_root',
        type=str,
        default='checkpoint',
        help='save root')

    args = parser.parse_args()
    main(args)
