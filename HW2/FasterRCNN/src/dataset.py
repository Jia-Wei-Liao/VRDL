import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class SVHN_Dataset(Dataset):
    def __init__(self, folder, data, transforms=None):
        self.data = data
        self.folder = folder
        self.transforms = transforms

    def __len__(self):
        return self.data['img_name'].unique().size

    def __getitem__(self, idx):
        image_name = self.data['img_name'].unique()[idx]
        image_path = os.path.join(self.folder, image_name)
        image = Image.open(image_path).convert("RGB")
        target = {
            'image_id': torch.tensor([int(image_name[:-4])]),
            'boxes': [],
            'labels': [],
            'area': [],
            'iscrowd': []
        }
        same_data = self.data[self.data['img_name'] == image_name]

        for i in range(same_data.shape[0]):
            x1 = same_data.iloc[i]['left']
            x2 = same_data.iloc[i]['right']
            y1 = same_data.iloc[i]['top']
            y2 = same_data.iloc[i]['bottom']
            target['boxes'].append([x1, y1, x2, y2])
            target['labels'].append([same_data.iloc[i]['label'].astype('int')])
            target['area'].append([(x2-x1)*(y2-y1)])
            target['iscrowd'].append([0])

        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        target['area'] = torch.as_tensor(target['area'], dtype=torch.float32)
        target['iscrowd'] = torch.as_tensor(target['iscrowd'],
                                            dtype=torch.int64)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
