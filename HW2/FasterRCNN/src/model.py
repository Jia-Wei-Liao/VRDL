import os
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNN(nn.Module):
    def __init__(self, num_classes, fine_tune=False):
        super(FasterRCNN, self).__init__()
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)

        if fine_tune:
            for param in model.parameters():
                param.requires_grad = False

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
                                             in_features, num_classes)

    def load(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path),
                                   map_location=torch.device('cpu'))

        return None

    def save(self, save_root, ep):
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, f'epoch={ep}.pth')
        torch.save(self.model.state_dict(), save_path)

        return None

    def forward(self, x, y=None):
        if y:
            return self.model(x, y)

        else:
            return self.model(x)
