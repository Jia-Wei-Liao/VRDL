import os
import random
import numpy as np
import torch
from argparse import Namespace
from src.datamodule import DataModule
from src.model import ResNet
from src.trainer import Trainer


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


params = Namespace()
params.mode = 'training'
params.model = 'resnext101'
params.fine_tune = False
params.max_epochs = 100
params.batch_size = 20
params.train_num = 2400
params.valid_num = 600
params.optimizer = 'AdamW'
params.lr = 1e-4
params.lr_scheduler = 'step'
params.lr_decay_period = 3
params.lr_decay_factor = 0.8
params.weight_decay = 1e-4
params.resize = (375, 375)
params.rand_noise_sigma = 0.02
params.rand_noise_prob = 0.1
params.baseline = 0.8
params.K_fold = True
params.valid_fold = 1
params.train_file = ["training_images"]
params.valid_file = ["training_images"]
params.test_file = ["testing_images"]
params.file_root = '/data/S/LinGroup/Users/sam/VRDL_HW1'
params.data_root = os.path.join(params.file_root, 'data')
params.save_path = os.path.join(params.file_root, 'checkpoint')


if __name__ == '__main__':
    if params.K_fold:
        for i in range(1, 5+1):
            print(f"Fold {i}:")
            params.valid_fold = i
            dataset = DataModule(params)
            model = ResNet(params)
            trainer = Trainer(params)
            trainer.fit(model, dataset)

            del dataset, model, trainer

    else:
        dataset = DataModule(params)
        model = ResNet(params)
        trainer = Trainer(params)
        trainer.fit(model, dataset)
