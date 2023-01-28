import os
import time
import torch
import torch.nn as nn
from src.utils import *


class Trainer():
    def __init__(self, params):
        self.params = params

    def configure_device(self):
        try:
            self.device = torch.device('cuda:0')
            print(f"Use {torch.cuda.device_count()} gpus.")

        except:
            self.device = torch.device('cpu:0')
            print("Use cpu.")

        return None

    def configure_loss_func(self):
        self.criterion = nn.CrossEntropyLoss()

        return None

    def configure_optimizers(self):
        Optimizers = {
          'Adam': torch.optim.Adam(
              self.model.parameters(),
              lr=self.params.lr,
              weight_decay=self.params.weight_decay
              ),
          'AdamW': torch.optim.AdamW(
              self.model.parameters(),
              lr=self.params.lr,
              weight_decay=self.params.weight_decay
              )
        }
        self.optimizer = Optimizers[self.params.optimizer]

        LR_Schedulers = {
          'step': torch.optim.lr_scheduler.StepLR(
              optimizer=self.optimizer,
              step_size=self.params.lr_decay_period,
              gamma=self.params.lr_decay_factor),
          'cos':  torch.optim.lr_scheduler.CosineAnnealingLR(
              optimizer=self.optimizer,
              T_max=4,
              eta_min=1e-6,
              last_epoch=-1
              )
        }
        self.lr_scheduler = LR_Schedulers[self.params.lr_scheduler]

        return None

    def compute_loss(self, x, y):
        loss = self.criterion(x, y)

        return loss

    def compute_acc(self, x, y):
        acc = torch.sum(torch.argmax(x, dim=1) == y)

        return acc

    def save_ckpt(self, ep, acc):
        save_name = f"fold={self.params.valid_fold}-ep={ep:0>3}-acc={acc:.4f}"
        self.model.save(save_name)

        return None

    def training_step(self):
        total_loss, total_acc = 0, 0
        self.model.train()
        for batch_data in self.train_loader:
            self.optimizer.zero_grad()

            images = batch_data['image'].to(self.device)
            labels = batch_data['label'].to(self.device)
            preds = self.model(images)
            losses = self.compute_loss(preds, labels)
            acc = self.compute_acc(preds, labels)

            losses.backward()
            self.optimizer.step()
            total_loss += losses*images.shape[0]
            total_acc += acc

            del images, labels, preds

        self.lr_scheduler.step()

        return {'loss': total_loss/self.params.train_num,
                'acc': total_acc/self.params.train_num}

    def validation_step(self):
        total_loss, total_acc = 0, 0
        self.model.eval()
        with torch.no_grad():
            for batch_data in self.valid_loader:
                images = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                preds = self.model(images)
                losses = self.compute_loss(preds, labels)
                acc = self.compute_acc(preds, labels)
                total_loss += losses*images.shape[0]
                total_acc += acc

                del images, labels, preds

        return {'loss': total_loss / self.params.valid_num,
                'acc': total_acc / self.params.valid_num}

    def test_step(self):
        classes_list = read_classes(os.path.join(
                                    self.params.data_root, 'classes.txt'))

        self.model.eval()
        with open(os.path.join(
                  self.params.data_root, 'answer.txt'), 'w') as fp:

            for batch_data in self.test_loader:
                images = batch_data['image'].to(self.device)
                preds = self.model(images).argmax(dim=1)
                for Id, pred in zip(batch_data['id'], preds):
                    fp.write(f"{Id} {classes_list[pred]}\n")
                    print(f"{Id} {classes_list[pred]}")

        return None

    def fit(self, model, dataset):
        self.configure_device()
        self.model = model.to(self.device)
        self.train_loader = dataset.train_dataloader()
        self.valid_loader = dataset.valid_dataloader()
        self.configure_loss_func()
        self.configure_optimizers()

        tic = time.time()
        for ep in range(1, self.params.max_epochs+1):
            train_record = self.training_step()
            print(f"epoch: {ep}/{self.params.max_epochs},",
                  f"time: {int(time.time() - tic)},",
                  f"type: train,",
                  f"loss: {train_record['loss']:.4f},",
                  f"acc: {train_record['acc']:.4f},",
                  f"lr: {self.lr_scheduler.get_last_lr()[0]:.8f}"
                  )

            valid_record = self.validation_step()
            print(f"epoch: {ep}/{self.params.max_epochs},",
                  f"time: {int(time.time() - tic)},",
                  f"type: valid,",
                  f"loss: {valid_record['loss']:.4f},",
                  f"acc: {valid_record['acc']:.4f}"
                  )

            if valid_record['acc'] > self.params.baseline:
                self.save_ckpt(ep=ep, acc=valid_record['acc'])

        return None

    def inference(self, model, dataset):
        self.configure_device()
        self.model = model.to(self.device)
        self.test_loader = dataset.test_dataloader()
        self.test_step()

        return None

    def fliplr(self, inputs):
        batch_size = inputs.shape[0]
        fliplr_tensor = torch.stack([torch.fliplr(
                        torch.flip(inputs[i, :, :, :], dims=[1, 2]))
                        for i in range(batch_size)], dim=0)

        return fliplr_tensor

    def ensemble(self, ModelList, dataset):
        self.configure_device()
        self.ModelList = [m.to(self.device).eval() for m in ModelList]
        self.test_loader = dataset.test_dataloader()
        classes_list = read_classes(os.path.join(
                                    self.params.data_root, 'classes.txt'))

        with open(os.path.join(
                  self.params.file_root, 'answer.txt'), 'w') as fp:

            with torch.no_grad():
                for batch_data in self.test_loader:

                    images = batch_data['image'].to(self.device)
                    fliplr_images = self.fliplr(images)

                    preds = torch.sum(torch.stack([m(images)+m(fliplr_images)
                                      for m in self.ModelList]),
                                      dim=0).argmax(dim=1)

                    for Id, pred in zip(batch_data['id'], preds):
                        fp.write(f"{Id} {classes_list[pred]}\n")
                        print(f"{Id} {classes_list[pred]}")

                    del images, preds, fliplr_images

        return None
