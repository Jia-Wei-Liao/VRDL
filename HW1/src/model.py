import os
import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101, resnext50_32x4d, resnext101_32x8d


class ResNet(nn.Module):
    def __init__(self, params, ckpt=False):
        super(ResNet, self).__init__()
        self.params = params
        self.model_dict = {
          'resnet50':  resnet50(pretrained=True),
          'resnet101': resnet101(pretrained=True),
          'resnext50': resnext50_32x4d(pretrained=True),
          'resnext101': resnext101_32x8d(pretrained=True),
        }
        self.model = self.model_dict[params.model]
        self.ckpt = ckpt

        self.modify_final_layer()
        self.use_data_parallel()

    def modify_final_layer(self):
        if self.params.fine_tune:
            for p in self.model.parameters():
                p.requires_grad = False
            print("Use fine tune.")

        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features,
            out_features=200,
            bias=True)

        if self.ckpt is not False:
            self.load(self.ckpt)

        return None

    def use_data_parallel(self):
        if self.params.mode == 'training':
            self.model = nn.DataParallel(self.model)

        return None

    def load(self, ckpt):
        print(f"load {ckpt}.")
        ckpt = torch.load(os.path.join(
            self.params.save_path, ckpt),
            map_location=torch.device('cpu'))
        self.model.load_state_dict(ckpt)

        return None

    def save(self, save_name):
        os.makedirs(self.params.save_path, exist_ok=True)
        torch.save(self.model.module.state_dict(),
                   os.path.join(self.params.save_path, save_name))

        return None

    def forward(self, inputs):
        outputs = self.model(inputs)

        return outputs
