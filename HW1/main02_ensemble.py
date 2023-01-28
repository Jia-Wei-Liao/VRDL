import os
from argparse import Namespace
from src.datamodule import DataModule
from src.model import ResNet
from src.trainer import Trainer


params = Namespace()
params.mode = 'inference'
params.model = 'resnext101'
params.fine_tune = False
params.batch_size = 20
params.resize = (375, 375)
params.train_file = ["training_images"]
params.valid_file = ["training_images"]
params.test_file = ["testing_images"]
params.file_root = '/data/S/LinGroup/Users/sam/VRDL_HW1'
params.data_root = os.path.join(params.file_root, 'data')
params.save_path = os.path.join(params.file_root, 'checkpoint')


if __name__ == '__main__':
    dataset = DataModule(params)
    ModelList = [
      ResNet(params, ckpt="fold=1-ep=019-acc=0.8333"),
      ResNet(params, ckpt="fold=1-ep=047-acc=0.8333"),
      ResNet(params, ckpt="fold=1-ep=054-acc=0.8383"),
      ResNet(params, ckpt="fold=2-ep=022-acc=0.8267"),
      ResNet(params, ckpt="fold=2-ep=033-acc=0.8267"),
      ResNet(params, ckpt="fold=2-ep=036-acc=0.8267"),
      ResNet(params, ckpt="fold=3-ep=029-acc=0.8067"),
      ResNet(params, ckpt="fold=3-ep=031-acc=0.8017"),
      ResNet(params, ckpt="fold=3-ep=044-acc=0.8050"),
      ResNet(params, ckpt="fold=4-ep=032-acc=0.8233"),
      ResNet(params, ckpt="fold=4-ep=040-acc=0.8233"),
      ResNet(params, ckpt="fold=4-ep=048-acc=0.8267"),
      ResNet(params, ckpt="fold=5-ep=043-acc=0.8250"),
      ResNet(params, ckpt="fold=5-ep=047-acc=0.8250"),
      ResNet(params, ckpt="fold=5-ep=049-acc=0.8233"),
    ]
    trainer = Trainer(params)
    trainer.ensemble(ModelList, dataset)
