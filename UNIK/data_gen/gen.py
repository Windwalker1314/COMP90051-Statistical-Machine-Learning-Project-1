import argparse
import pickle
from tqdm import tqdm
import sys

sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization

max_body_true = 1
num_joint = 20
max_frame = 16

import os
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import math

def aug(args):
    all_data = pd.read_csv(os.path.join(args.data_path, 'train.csv'), sep=',', header = None)
    label2id = defaultdict(set)
    for i in range(len(all_data)):
        label2id[all_data.iloc[i,-1]].add(all_data.iloc[i,0]-1)
    valid_id = []
    train_id = []
    for k,v in label2id.items():
        for sample_id in list(v)[:5]:
            valid_id.append(sample_id)
        for sample_id in list(v)[5:]:
            train_id.append(sample_id)

    train = all_data.iloc[train_id,:]
    val = all_data.iloc[valid_id,:]
    if args.aug:
        print ('apply augmentation')
        label2id = defaultdict(set)
        for i in range(len(train)):
            label2id[train.iloc[i,-1]].add(train.iloc[i,0]-1)
        labels_count = Counter(train.iloc[:,-1])
        aug_data = train.iloc[list(label2id[41])[0]-40*5:,:].copy()
        aug_data.iloc[:,0] = aug_data.iloc[:,0] + len(aug_data)
        # A,B = aug_data.iloc[:,1:-1].shape
        # noise = np.random.normal(0,0.05,size=(A,B))
        # aug_data.iloc[:,1:-1] = aug_data.iloc[:,1:-1] + noise
        new_data = train.append(aug_data,ignore_index = True)

        # aug_data = train.iloc[list(label2id[41])[0]:,:].copy()
        # aug_data.iloc[:,0] = aug_data.iloc[:,0] + len(new_data) - list(label2id[41])[0] + 1
        # A,B = aug_data.iloc[:,1:-1].shape
        # noise = np.random.normal(0,0.01,size=(A,B))
        # aug_data.iloc[:,1:-1] = aug_data.iloc[:,1:-1] + noise
        # new_data = new_data.append(aug_data,ignore_index = True)
        new_data.to_csv(os.path.join(args.data_path, 'train_aug.csv'), index=False,header=False)
    else:
        train.to_csv(os.path.join(args.data_path, 'train_aug.csv'), index=False,header=False)

    val.to_csv(os.path.join(args.data_path, 'val_aug.csv'), index=False,header=False)

def gendata(data_path, out_path, part='eval'):

    if part == 'train':
        with open(os.path.join(data_path, 'train_aug.csv')) as f:
            all_data = f.read().splitlines()
    elif part == 'val':
        with open(os.path.join(data_path, 'val_aug.csv')) as f:
            all_data = f.read().splitlines()
    else:
        with open(os.path.join(data_path, 'test.csv')) as f:
            all_data = f.read().splitlines()
    sample_name = list(map(lambda x: x.split(',')[0], all_data))
    if part == 'test':
        sample_label = [0]*len(all_data)
    else:
        sample_label = list(map(lambda x: int(x.split(',')[-1])-1, all_data))
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    data = np.zeros((len(all_data), 1, max_frame, num_joint, 3))
    for sample_id, frame_data in enumerate(all_data):
        frame_data = list(map(lambda x: float(x), frame_data.split(',')[1:]))
        idx = 0
        for i in range(max_frame):
            for j in range(num_joint):
                for k in range(3):
                    data[sample_id, 0, i, j, k] = frame_data[idx]
                    idx += 1
    fp = data.transpose(0, 4, 2, 3, 1)
    fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kaggle Data Converter.')
    parser.add_argument('--data_path', default='../data/kaggle_raw/')
    parser.add_argument('--out_folder', default='../data/kaggle/')
    parser.add_argument('--aug', default=True)

    part = ['train', 'val','test']
    args = parser.parse_args()

    aug(args)

    for p in part:
        out_path = args.out_folder
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        gendata(
            args.data_path,
            out_path,
            part=p)
