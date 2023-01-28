import os
import glob
import tqdm
import imageio
import argparse
import numpy as np
from PIL import Image


def get_LR_image(HR_image, scale):
    h, w = HR_image.shape[:2]
    h /= scale
    w /= scale
    LR_image = np.array(
        Image.fromarray(HR_image).resize((int(w), int(h)), Image.BICUBIC)
        )

    return LR_image


def clip(image, scale):
    channel = len(image.shape)
    h, w = image.shape[:2]
    h -= np.mod(h, scale)
    w -= np.mod(w, scale)

    if channel == 3:
        clip_image = image[0:h, 0:w, :]

    else:
        clip_image = image[0:h, 0:w]

    return clip_image


def main(args):
    data_list = os.listdir(os.path.join(args.dataset))
    HR_root = os.path.join(args.save_root, f'{args.mode}_HR_x{args.scale}')
    LR_root = os.path.join(args.save_root, f'{args.mode}_LR_x{args.scale}')

    os.makedirs(HR_root, exist_ok=True)
    os.makedirs(LR_root, exist_ok=True)

    for i, image_name in tqdm.tqdm(enumerate(data_list)):
        image = imageio.imread(os.path.join(args.dataset, image_name))
        HR_image = clip(image, args.scale)
        LR_image = get_LR_image(HR_image, args.scale)

        imageio.imwrite(os.path.join(HR_root, image_name), HR_image)
        imageio.imwrite(os.path.join(LR_root, image_name), LR_image)

    print('finish!')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,
                        default='val',
                        help='prepare training set')
    parser.add_argument('--dataset', type=str,
                        default='dataset/val_HR_data',
                        help='dataset')
    parser.add_argument('--save_root', type=str,
                        default='dataset',
                        help='save root')
    parser.add_argument('--augment', type=bool,
                        default=False,
                        help='use data augmentation')
    parser.add_argument('--scale', type=int,
                        default=3,
                        help='scale')
    args = parser.parse_args()

    main(args)
