# -*- coding: utf-8 -*-

"""
Modify from github: https://github.com/kayoyin/digit-detector/blob/master/construct_data.py
@author: PavitrakumarPC
"""

import os
import h5py
import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument('--train_num',
    type=int,
    default=30000,
    help='number of training data')

parser.add_argument('--valid_num',
    type=int,
    default=3402,
    help='number of validation data')

args = parser.parse_args()


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])


def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs[key] = values
        
    return attrs


def img_boundingbox_data_constructor(start, end, mat_file):
    f = h5py.File(mat_file,'r') 
    all_rows = []
    bbox_df = pd.DataFrame([],columns=['height','img_name','label','left','top','width'])
    
    for j in range(start, end):
        print(f'step: {j}')
        img_name = get_name(j, f)
        row_dict = get_bbox(j, f)
        row_dict['img_name'] = img_name
        all_rows.append(row_dict)
        bbox_df = pd.concat([bbox_df, pd.DataFrame.from_dict(row_dict, orient = 'columns')])
    
    bbox_df['bottom'] = bbox_df['top'] + bbox_df['height']
    bbox_df['right'] = bbox_df['left'] + bbox_df['width']

    return bbox_df


def train_to_csv(img_folder, mat_file_name, train_num, valid_num):
    train_data = img_boundingbox_data_constructor(
        0, train_num, os.path.join(img_folder, mat_file_name))

    valid_data = img_boundingbox_data_constructor(
        train_num, train_num + valid_num, os.path.join(img_folder, mat_file_name))
    
    train_data.to_csv('train_data.csv', index=False)
    valid_data.to_csv('valid_data.csv', index=False)


if __name__ == '__main__':
    train_to_csv('./train', 'digitStruct.mat', args.train_num, args.valid_num)
