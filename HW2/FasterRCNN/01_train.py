import os
import argparse
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from detection.engine import train_one_epoch, evaluate

from src.model import FasterRCNN
from src.dataset import SVHN_Dataset
from src.transforms import train_transform
from src.utils import *


cur_time = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
parser = argparse.ArgumentParser()

parser.add_argument(
    '--save_root',
    type=str,
    default=f'./checkpoint/{cur_time}',
    help='save root'
    )

parser.add_argument(
    '--folder',
    type=str,
    default='./train',
    help='train image folder'
    )

parser.add_argument(
    '--optimizer',
    type=str,
    default='AdamW',
    help='optimizer')

parser.add_argument(
    '--lr',
    type=float,
    default=1e-4,
    help='initial_learning_rate'
    )

parser.add_argument(
    '--weight_decay',
    type=float,
    default=5e-4,
    help='weight decay'
    )

parser.add_argument(
    '--step_size',
    type=int,
    default=1,
    help='learning decay period'
    )

parser.add_argument(
    '--gamma',
    type=float,
    default=0.9,
    help='learning rate decay factor'
    )

parser.add_argument(
    '--max_epochs',
    type=int,
    default=20,
    help='maximum epochs'
    )

parser.add_argument(
    '--batch_size',
    type=int,
    default=4,
    help='batch size'
    )

parser.add_argument(
    '--num_workers',
    type=int,
    default=4,
    help='number of workers'
    )

parser.add_argument(
    '--num_classes',
    type=int,
    default=11,
    help='number of classes'
    )

parser.add_argument(
    '--cuda',
    type=int,
    default=0,
    help='cuda'
    )

args = parser.parse_args()


if __name__ == '__main__':
    train_data = pd.read_csv('train_data.csv')
    valid_data = pd.read_csv('valid_data.csv')
    train_set = SVHN_Dataset(args.folder, train_data, train_transform)
    valid_set = SVHN_Dataset(args.folder, valid_data, train_transform)
    print(f'train: {len(train_set)}, valid: {len(valid_set)}')

    train_dataloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

    model = FasterRCNN(args.num_classes)
    device = get_device(args.cuda)
    model.to(device)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
            )

    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
            )

    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
            )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma)

    model.save(args.save_root, 0)
    save_json_file(vars(args), os.path.join(args.save_root, 'config.json'))

    for ep in range(1, args.max_epochs+1):
        train_one_epoch(
            model,
            optimizer,
            train_dataloader,
            device,
            ep,
            print_freq=10
        )
        lr_scheduler.step()

        with torch.no_grad():
            evaluate(model, valid_dataloader, device=device)

        model.save(args.save_root, ep)
