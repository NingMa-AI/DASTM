# coding=utf-8
from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import numpy as np
import shutil
import errno
import torch
import os
import pickle
import random
import gl

'''
Inspired by https://github.com/pytorch/vision/pull/46
'''

IMG_CACHE = {}

class NTU_RGBD_Dataset(data.Dataset):

    def __init__(self, mode='train', data_list=None, debug=False, extract_frame=1, transform=None, target_transform=None):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        '''
        super(NTU_RGBD_Dataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform


        if gl.dataset == 'ntu120_30':
            path="********************************to be specified********************************"
            segment = 30
    

        print('data_path :{}'.format(path))



        if mode == 'train':
            data_path = os.path.join(path, 'train_data.npy')
            label_path = os.path.join(path, 'train_label.npy')
            num_frame = os.path.join(path, 'train_frame.npy')
        elif mode == 'val':
            data_path = os.path.join(path, 'val_data.npy')
            label_path = os.path.join(path, 'val_label.npy')
            num_frame = os.path.join(path, 'val_frame.npy')
        else:
            data_path = os.path.join(path, 'test_data.npy')
            label_path = os.path.join(path, 'test_label.npy')
            num_frame = os.path.join(path, 'test_frame.npy')

        self.data, self.label, self.num_frame = np.load(data_path), np.load(label_path), np.load(num_frame)

        # print('min = ', np.min(self.data), ' max = ', np.max(self.data))

        if debug:
            data_len = len(self.label)
            data_len = int(0.1 * data_len)
            self.label = self.label[0:data_len]
            self.data = self.data[0:data_len]
            self.num_frame = self.num_frame[0:data_len]

        if extract_frame == 1:
            self.data = self.extract_frame(self.data, self.num_frame, segment)

        print('sample_num in {}'.format(mode), len(self.label))
        n_classes = len(np.unique(self.label))
        print('n_class', n_classes)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.label[idx]

    def __len__(self):
        return len(self.label)

    def extract_frame(self, x, num_frame, segment):
        n, c, t, v, m = x.shape
        assert n == len(num_frame)

        num_frame = np.array(num_frame)
        step = num_frame // segment
        new_x = []

        for i in range(n):
            if num_frame[i] < segment:
                new_x.append(np.expand_dims(x[i, :, 0:segment, :, :], 0).reshape(1, c, segment, v, m))
                continue
            idx = [random.randint(j * step[i], (j + 1) * step[i] - 1) for j in range(segment)]
            new_x.append(np.expand_dims(x[i, :, idx, :, :], 0).reshape(1, c, segment, v, m))

        new_x = np.concatenate(new_x, 0)
        return new_x
