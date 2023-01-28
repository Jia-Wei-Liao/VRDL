import copy
import torch
import random
import numpy as np
import detectron2.data.transforms as T
from detectron2.data import detection_utils
from detectron2.data.build import build_detection_train_loader
from detectron2.engine.defaults import DefaultTrainer


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        dataloader = build_detection_train_loader(cfg, mapper=custom_mapper)

        return dataloader


def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = detection_utils.read_image(dataset_dict['file_name'], format='BGR')
    aug_input = T.StandardAugInput(image)
    aug_transform = [
        T.RandomBrightness(0.9, 1.1),
        T.RandomCrop('relative', (0.5, 0.5)),
        T.ResizeShortestEdge(
            short_edge_length=608,
            max_size=800,
            sample_style='choice'),
        T.RandomFlip(prob=0.5)
    ]
    transforms = aug_input.apply_augmentations(aug_transform)
    image = aug_input.image
    image_shape = image.shape[:2]
    dataset_dict['image'] = torch.as_tensor(
        np.ascontiguousarray(image.transpose(2, 0, 1)))

    annos = [
        detection_utils.transform_instance_annotations(
            annotation,
            transforms,
            image_shape)
        for annotation in dataset_dict.pop('annotations')
    ]

    instances = detection_utils.annotations_to_instances(
        annos, image_shape, mask_format='bitmask'
    )

    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
    dataset_dict["instances"] = detection_utils.filter_empty_instances(
                                instances)

    return dataset_dict
