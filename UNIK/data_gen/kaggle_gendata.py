import argparse
import pickle
from tqdm import tqdm
import sys

sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization

max_body_true = 1
num_joint = 20
max_frame = 16

import numpy as np
import os

with open('../data/kaggle_raw/train.csv') as f:
    all_data = f.readlines()

from collections import defaultdict
label2id = defaultdict(set)
for line in all_data:
    line = line.split(',')
    label2id[int(line[-1])].add(int(line[0]))

import math
valid_id = set()
for k,v in label2id.items():
    for sample_id in list(v)[:math.ceil(len(v)/10)]:
        valid_id.add(sample_id)

def gendata(data_path, out_path, valid_id, benchmark='xview', part='eval'):
    with open(os.path.join(data_path, 'train.csv')) as f:
        all_data = f.readlines()

    if part == 'train':
        all_data = list(filter(lambda x: not int(x.split(',')[0]) in valid_id, all_data))
    elif part == 'val':
        all_data = list(filter(lambda x: int(x.split(',')[0]) in valid_id, all_data))
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

    data = np.zeros((len(all_data), 2, max_frame, num_joint, 3))
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

    benchmark = ['xsub', 'xview']
    part = ['train', 'val','test']
    args = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(args.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(
                args.data_path,
                out_path,
                valid_id,
                benchmark=b,
                part=p)
