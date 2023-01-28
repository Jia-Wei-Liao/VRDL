import os
import random
import os
import tqdm
import imageio
import argparse


def main(args):
    train_root = os.path.join(args.save_root, 'train_HR_data')
    val_root = os.path.join(args.save_root, 'val_HR_data')
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)

    data_list = os.listdir(os.path.join(args.dataset))
    random.shuffle(data_list)
    train_num = int(len(data_list)*args.ratio)
    train_data_list = data_list[:train_num]
    val_data_list = data_list[train_num:]

    for i, image_name in tqdm.tqdm(enumerate(train_data_list)):
        image = imageio.imread(os.path.join(args.dataset, image_name))
        imageio.imwrite(os.path.join(train_root, image_name), image)

    for i, image_name in tqdm.tqdm(enumerate(val_data_list)):
        image = imageio.imread(os.path.join(args.dataset, image_name))
        imageio.imwrite(os.path.join(val_root, image_name), image)

    print('finish!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='dataset/training_hr_images',
                        help='dataset')
    parser.add_argument('--save_root', type=str,
                        default='dataset',
                        help='save root')
    parser.add_argument('--ratio', type=float,
                        default=0.983,
                        help='training number over validation number')
    args = parser.parse_args()

    main(args)
