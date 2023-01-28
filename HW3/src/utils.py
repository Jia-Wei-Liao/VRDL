import os
import json
import yaml
from datetime import datetime
from detectron2 import model_zoo
from detectron2.config import get_cfg


def read_json(json_path):
    with open(json_path, 'r') as fp:
        data_dict = json.load(fp)

    return data_dict


def save_json(save_list, save_path):
    with open(save_path, 'w') as fp:
        json.dump(save_list, fp, indent=4)

    return None


def save_config(cfg, save_path):
    with open(save_path, 'w') as fp:
        yaml.dump(cfg, fp, default_flow_style=False)

    return None


def get_save_path(save_root):
    cur_time = datetime.today().strftime('%Y-%m-%d-%H-%M')
    save_path = os.path.join(save_root, cur_time)
    os.makedirs(save_path, exist_ok=True)

    return save_path


def set_cfg(args):
    cfg = get_cfg()

    # Set the model
    model_path = os.path.join(
        'COCO-InstanceSegmentation', f'{args.model}.yaml')
    cfg.merge_from_file(model_zoo.get_config_file(model_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    if 'C4' in args.model:
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]

    elif 'FPN' in args.model:
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]

    # Set the customer dataset
    cfg.DATASETS.TRAIN = ("Nuclei_data", )
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1

    # Set the optimizer
    iter_one_epoch = int(args.train_num / args.batch_size)
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.SOLVER.MAX_ITER = args.epoch * iter_one_epoch
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 10000
    cfg.SOLVER.WARMUP_ITERS = args.warmup_ep * iter_one_epoch
    cfg.SOLVER.GAMMA = args.decay_factor
    cfg.SOLVER.STEPS = tuple([s*iter_one_epoch for s in args.lr_decay_ep])
    cfg.SOLVER.CHECKPOINT_PERIOD = 5 * iter_one_epoch

    # Set for the inference step
    cfg.INPUT.MIN_SIZE_TEST = 1000
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.TEST.EVAL_PERIOD = 0
    cfg.TEST.DETECTIONS_PER_IMAGE = 500
    cfg.TEST.AUG["ENABLED"] = True
    cfg.TEST.AUG.MIN_SIZES = (1500, 1600, 1700)

    cfg.MODEL.DEVICE = args.device
    cfg.OUTPUT_DIR = get_save_path(args.save_root)
    save_config(cfg, os.path.join(cfg.OUTPUT_DIR, 'config.yaml'))

    return cfg
