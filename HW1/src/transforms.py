import random
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    RandomHorizontalFlip,
    RandomRotation,
    RandomAffine,
    Normalize
    )


class LoadImg():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = Image.open(data[key])

            else:
                raise KeyError(f"{key} is not a key of data.")

        return data


class ResizeImg():
    def __init__(self, keys, size=(224, 224)):
        self.keys = keys
        self.size = size

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = Resize(size=self.size)(data[key])

            else:
                raise KeyError(f"{key} is not a key of data.")

        return data


class RandomTrans():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = RandomHorizontalFlip()(data[key])
                data[key] = RandomRotation(15)(data[key])
                data[key] = RandomAffine(0, shear=10,
                                         scale=(0.8, 1.2))(data[key])

            else:
                raise KeyError(f"{key} is not a key of data.")

        return data


class ImgToTensor():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = ToTensor()(data[key])

            else:
                raise KeyError(f"{key} is not a key of data.")

        return data


class ImgNormalize():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = Normalize((0.485, 0.456, 0.406),
                                      (0.229, 0.224, 0.225))(data[key])

            else:
                raise KeyError(f"{key} is not a key of data.")

        return data


class RandomNoise():
    def __init__(self, keys, p=0.1, sigma=0.01):
        self.keys = keys
        self.prob = p
        self.sigma = sigma

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                if random.random() <= self.prob:
                    data[key] += self.sigma * torch.randn(data[key].shape)

            else:
                raise KeyError(f"{key} is not a key of data.")

        return data


class GridMask():
    def __init__(self, keys, p=0.25, dmin=60, dmax=160, ratio=0.6):
        self.keys = keys
        self.prob = p
        self.dmin = dmin
        self.dmax = dmax
        self.ratio = ratio

    def generate_grid_mask(self, image):
        d = random.randint(self.dmin, self.dmax)
        dx, dy = random.randint(0, d-1), random.randint(0, d-1)
        sl = int(d * (1-self.ratio))
        for i in range(dx, image.shape[0], d):
            for j in range(dy, image.shape[1], d):
                row_end = min(i+sl, image.shape[0])
                col_end = min(j+sl, image.shape[1])
                image[:, i:row_end, j:col_end] = 0

        return image

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                if random.random() <= self.prob:
                    data[key] = self.generate_grid_mask(data[key])

            else:
                raise KeyError(f"{key} is not a key of data.")

        return data
