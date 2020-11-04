# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp
import os

from .bases import BaseImageDataset
from sklearn.model_selection import train_test_split

class BREAK(Exception): pass

class sig(BaseImageDataset):
    dataset_dir = 'sig'

    def __init__(self, root='./data', verbose=True, **kwargs):
        super(sig, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train, self.gallery, self.query = self.create_dataset(self.dataset_dir)

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs = self.get_imagedata_info(self.gallery)

    def create_dataset(self, data_path, train_ratio=0.6, gallery_ratio=0.28):
        heap = []
        data_size = 0
        for index, folder in enumerate(os.listdir(data_path)):
            img_set = set()
            folder_path = os.path.join(data_path, folder)
            for img in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img)
                img_set.add((img_path,index))
                data_size += 1
            if len(img_set) != 0:
                heap.append(img_set)
                
        train_size = int(data_size*train_ratio)
        gallery_size = int(data_size*gallery_ratio)
        
        train = []
        gallery = []
        while True:
            try:
                for img_set in heap:
                    if len(img_set) == 0:
                        heap.remove(img_set)
                    else:
                        train.append(img_set.pop())
                        if len(train) == train_size:
                            raise BREAK
            except BREAK:
                break
            
        while True:
            try:
                for img_set in heap:
                    if len(img_set) == 0:
                        heap.remove(img_set)
                    else:
                        gallery.append(img_set.pop())
                        if len(gallery) == gallery_size:
                            raise BREAK
            except BREAK:
                break
                
        query = set()
        for img_set in heap:
            if len(img_set) != 0:
                query.update(img_set)
        query = list(query)
        
        return train, gallery, query
    
if __name__ == "__main__":
    s = sig()