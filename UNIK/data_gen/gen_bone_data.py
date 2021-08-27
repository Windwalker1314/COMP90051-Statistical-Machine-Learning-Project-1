import os
import numpy as np
from numpy.lib.format import open_memmap

paris = {
    'ntu/xview': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
    'ntu/xsub': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),

    'ntu120/xview': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
    'ntu120/xsub': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),

    'smarthome/xsub': (
        (5, 4), (4, 3), (3, 2), (2, 1), (6, 3), (8, 6), (10, 8), (7, 3), (9, 7), (11, 9), (12, 5), (14, 12), (16, 14),
        (13, 5), (15, 13), (17, 15)
    ),
    'smarthome/xview1': (
        (5, 4), (4, 3), (3, 2), (2, 1), (6, 3), (8, 6), (10, 8), (7, 3), (9, 7), (11, 9), (12, 5), (14, 12), (16, 14),
        (13, 5), (15, 13), (17, 15)
    ),
    'smarthome/xview2': (
        (5, 4), (4, 3), (3, 2), (2, 1), (6, 3), (8, 6), (10, 8), (7, 3), (9, 7), (11, 9), (12, 5), (14, 12), (16, 14),
        (13, 5), (15, 13), (17, 15)
    ),

    'penn': (
        (5, 4), (4, 3), (3, 2), (2, 1), (6, 3), (8, 6), (10, 8), (7, 3), (9, 7), (11, 9), (12, 5), (14, 12), (16, 14),
        (13, 5), (15, 13), (17, 15)
    ),

    'posetics': (
        (5, 4), (4, 3), (3, 2), (2, 1), (6, 3), (8, 6), (10, 8), (7, 3), (9, 7), (11, 9), (12, 5), (14, 12), (16, 14),
        (13, 5), (15, 13), (17, 15)
    ),

    'kaggle/xsub':(
        (1, 2), (2, 3), (4, 3), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
        (19, 18), (20, 19), (5, 3),(6, 5), (7, 6), (8, 7), (9, 3), (10, 9), (11, 10),
        (12, 11), (3, 3)
          ),

}

sets = {
    'train', 'val', 'test'
}

# 'ntu/xview', 'ntu/xsub', 'posetics'
datasets = {
    'kaggle/xsub'
}
# bone
from tqdm import tqdm

for dataset in datasets:
    for set in sets:
        print(dataset, set)
        data = np.load('../data/{}/{}_data_joint.npy'.format(dataset, set))
        N, C, T, V, M = data.shape
        fp_sp = open_memmap(
            '../data/{}/{}_data_bone.npy'.format(dataset, set),
            dtype='float32',
            mode='w+',
            shape=(N, 3, T, V, M))

        fp_sp[:, :C, :, :, :] = data
        for v1, v2 in tqdm(paris[dataset]):
            if dataset != 'kinetics':
                v1 -= 1
                v2 -= 1
            fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]