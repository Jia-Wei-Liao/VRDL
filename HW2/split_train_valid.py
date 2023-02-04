import os
import glob
import random
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    '--valid_num',
    type=int,
    default=3402,
    help='number of validation data'
    )

parser.add_argument(
    '--train_dir',
    type=str,
    default='train',
    help='data_location'
    )

parser.add_argument(
    '--valid_dir',
    type=str,
    default='valid',
    help='valid_data_location'
    )

args = parser.parse_args()


def get_id(name_list):
    id_list = list(map(lambda x: x.split('\\')[-1][:-4], name_list))

    return id_list


if __name__ == '__main__':
    list_dir = glob.glob(os.path.join(args.train_dir, '*.png'))
    id_list = get_id(list_dir)
    random.shuffle(id_list)
    valid_id = id_list[:args.valid_num]
    image_format = '{}.png'
    label_format = '{}.txt'
    os.makedirs(args.valid_dir, exist_ok=True)

    for i, image_id in enumerate(valid_id):
        os.rename(os.path.join(args.train_dir, image_format.format(image_id)),
                  os.path.join(args.valid_dir, image_format.format(image_id)))
        os.rename(os.path.join(args.train_dir, label_format.format(image_id)),
                  os.path.join(args.valid_dir, label_format.format(image_id)))

        print(f'[{i+1}/{len(valid_id)}]')
