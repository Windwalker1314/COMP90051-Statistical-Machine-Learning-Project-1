# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 04:23:30 2021

@author: Windwalker
"""
import pandas as pd
import numpy as np

# random split 
raw_data = pd.read_csv('train.csv',header=None,dtype='float32')
TRAIN_SIZE = 0.8
msk = np.random.rand(len(raw_data))<TRAIN_SIZE
train=raw_data[msk]
test =raw_data[~msk]
train.to_csv('training_set.csv',index=False,header=None)
test.to_csv('val_set.csv',index=False,header=None)

# Balanced validation set
df = pd.read_csv("train.csv",header=None)
test = df.groupby(961).sample(n=5)
test.to_csv('val_set_b.csv',index=False,header=None)
train = df.drop(test.index)
train.to_csv('training_set_b.csv',index=False,header=None)