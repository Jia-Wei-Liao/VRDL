import os
import numpy as np
import torch
from src.transforms import *
from torch.utils.data import Dataset, DataLoader


class Bird(Dataset):
    def __init__(self, params, Type, file_list, transform=None):
        self.params = params
        self.type = Type
        self.data_type = {
          'train': self.params.train_file,
          'valid': self.params.valid_file,
          'test':  self.params.test_file
        }
        self.file_list = [(file, data) for data in file_list
                          for file in self.data_type[Type]]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        if self.type == 'test':
            file, image = self.file_list[i]
            label = -1

        else:
            file, data = self.file_list[i]
            image, label = data[0], data[1][:3]

        data = {
          'id':    image,
          'image': os.path.join(self.params.data_root, file, image),
          'label': int(label)-1
        }

        if self.transform is not None:
            data = self.transform(data)

        return data


class DataModule():
    def __init__(self, params):
        self.params = params

    def read_file(self, file):
        with open(file, 'r') as f:
            content = f.readlines()

        return content

    def train_dataloader(self):
        if self.params.mode == 'training':
            file_list = []
            for i in range(1, 6):
                if i != self.params.valid_fold:
                    file_list += self.read_file(
                        os.path.join(self.params.data_root,
                                     'fold', f'fold{i}.txt'))

            data_list = [tuple(word.strip().split())
                         for word in file_list][:self.params.train_num]

            train_trans = Compose([
                LoadImg(keys=["image"]),
                RandomTrans(keys=["image"]),
                ResizeImg(keys=["image"], size=self.params.resize),
                ImgToTensor(keys=["image"]),
                RandomNoise(
                    keys=["image"],
                    p=self.params.rand_noise_prob,
                    sigma=self.params.rand_noise_sigma
                ),
                ImgNormalize(keys=["image"]),
                GridMask(keys=["image"], dmin=90, dmax=300, ratio=0.7, p=0.3)
            ])

            train_set = Bird(
                self.params,
                'train',
                data_list,
                transform=train_trans)

            train_loader = DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                num_workers=4, shuffle=True)
            print(f"train num: {len(train_set)}.")
            return train_loader

        else:
            return None

    def valid_dataloader(self):
        if self.params.mode == 'training':
            file_list = self.read_file(
                os.path.join(self.params.data_root,
                             'fold', f'fold{self.params.valid_fold}.txt'))

            data_list = [tuple(word.strip().split())
                         for word in file_list][:self.params.valid_num]

            valid_trans = Compose([
                LoadImg(keys=["image"]),
                ResizeImg(keys=["image"], size=self.params.resize),
                ImgToTensor(keys=["image"]),
                ImgNormalize(keys=["image"])
            ])

            valid_set = Bird(
                self.params,
                'valid',
                data_list,
                transform=valid_trans)

            valid_loader = DataLoader(
                valid_set,
                batch_size=self.params.batch_size,
                num_workers=4, shuffle=False)
            print(f"valid num: {len(valid_set)}.")
            return valid_loader

        else:
            return None

    def test_dataloader(self):
        if self.params.mode == 'inference':
            file_list = self.read_file(
                os.path.join(
                    self.params.data_root, 'testing_img_order.txt'))

            data_list = [word.strip() for word in file_list]

            test_trans = Compose([
                LoadImg(keys=["image"]),
                ResizeImg(keys=["image"], size=self.params.resize),
                ImgToTensor(keys=["image"]),
                ImgNormalize(keys=["image"])
            ])

            test_set = Bird(
                self.params,
                'test',
                data_list,
                transform=test_trans)

            test_loader = DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                num_workers=4, shuffle=False)
            print(f"test num: {len(test_set)}.")
            return test_loader

        else:
            return None
