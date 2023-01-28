import os
import tqdm
import imageio
import argparse
import options.options as option

from solvers import create_solver
from data import create_dataset, create_dataloader
from utils import util


def main(args):
    opt = option.parse(args.opt)
    opt = option.dict_to_nonedict(opt)
    solver = create_solver(opt)

    bm_names = []
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)
        bm_names.append(test_set.name())

    for bm, test_loader in zip(bm_names, test_loaders):
        save_path_list = opt['solver']['pretrained_path'].split(os.sep)[:-2]
        save_path = '/'.join(save_path_list)
        save_img_path = os.path.join(save_path, 'result')
        os.makedirs(save_img_path, exist_ok=True)

        for batch in tqdm.tqdm(test_loader):
            solver.feed_data(batch, need_HR=False)
            solver.test()
            visuals = solver.get_current_visual(need_HR=False)
            imageio.imwrite(os.path.join(
                save_img_path,
                os.path.basename(batch['LR_path'][0])[:-4]+'_pred.png'
                ), visuals['SR'])

    print("finish!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True,
                        help='path to options json file.')
    args = parser.parse_args()
    main(args)
