import torch.utils.data as data

from data import common


class LRDataset(data.Dataset):

    def name(self):
        return common.find_benchmark(self.opt['dataroot_LR'])

    def __init__(self, opt):
        super(LRDataset, self).__init__()
        self.opt = opt
        self.scale = self.opt['scale']
        self.paths_LR = None
        self.paths_LR = common.get_image_paths(
            opt['data_type'], opt['dataroot_LR'])

    def __getitem__(self, idx):
        lr, lr_path = self._load_file(idx)
        lr_tensor = common.np2Tensor([lr], self.opt['rgb_range'])[0]

        return {'LR': lr_tensor, 'LR_path': lr_path}

    def __len__(self):
        return len(self.paths_LR)

    def _load_file(self, idx):
        lr_path = self.paths_LR[idx]
        lr = common.read_img(lr_path, self.opt['data_type'])

        return lr, lr_path
