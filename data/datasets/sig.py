# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os

from .bases import BaseImageDataset
from sklearn.model_selection import train_test_split

import random 
import numpy as np

class BREAK(Exception): pass

class sig(BaseImageDataset):
    dataset_dir = 'sig'

    def __init__(self, root='./data', label_path="./label_signature.txt", forgery_recog=False, verbose=True, **kwargs):
        super(sig, self).__init__()
        # self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.dataset_dir = root

        if not forgery_recog:
            self.train, self.gallery, self.query = self.create_dataset(self.dataset_dir)

            self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
            self.num_query_pids, self.num_query_imgs = self.get_imagedata_info(self.query)
            self.num_gallery_pids, self.num_gallery_imgs = self.get_imagedata_info(self.gallery)

        else:
            if not os.path.exists(label_path):
                raise Exception("Label file {} is not existed. Please check !!")
            with open(label_path, "r") as f:
                label_data = f.readlines()
            
            self.train, self.gallery, self.query = self.create_dataset_with_forgery(self.dataset_dir, label_data)

            self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
            self.num_query_pids, self.num_query_imgs = self.get_imagedata_info(self.query)
            self.num_gallery_pids, self.num_gallery_imgs = self.get_imagedata_info(self.gallery)

    
    def create_dataset_with_forgery(self, root, label_data, train_ratio=0.7, gallery_ratio=0.3):
        dict_id = {}
        for line in label_data:
            image_name, idx, real = line.strip().split(" ")
            image_path = os.path.join(root, image_name)
            if idx not in dict_id.keys():
                dict_id.update({idx: [(image_path, int(idx), int(real))]})
            else:
                dict_id[idx].extend([(image_path, int(idx), int(real))])
            
        train = []
        gallery = []
        query = []
        
        id_label = 0
        for key in dict_id:
            id_set = dict_id[key]
            sub_train = random.sample(id_set, int(train_ratio * len(id_set)))  
            sub_query = list(set(id_set) - set(sub_train))
            sub_gallery = sub_query

            train.extend(sub_train)
            gallery.extend(sub_gallery)
            query.extend(sub_query)
        
        # print(train)
        return train, gallery, query


    def create_dataset(self, dataset_path, train_ratio=0.7, gallery_ratio=0.3):
        train = []
        gallery = []
        query = []


        all_id_folder = os.listdir(dataset_path)
        id_label = 0
        for index, id_folder in enumerate(all_id_folder):
            id_set = []
            id_folder_path = os.path.join(dataset_path, id_folder)
            # each_id_folder_path = glob.glob(os.path.join(id_folder_path, "*"))
            each_id_folder_path = glob.glob(os.path.join(id_folder_path, "*", "*"))
            if len(each_id_folder_path) == 0:
                continue
            for path in each_id_folder_path:
                id_set.append((path, id_label))
            
            sub_train = random.sample(id_set, int(train_ratio * len(id_set)))
            # sub_gallery = random.sample(id_set, int(gallery_ratio * len(id_set)))
            sub_query = list(set(id_set) - set(sub_train))
            sub_gallery = sub_query

            train.extend(sub_train)
            gallery.extend(sub_gallery)
            query.extend(sub_query)
            id_label += 1
        
        count_id_train = [0] *  id_label
        for each in train:
            count_id_train[each[1]] += 1
        
        count_id_gal = [0] *  id_label
        for each in gallery:
            count_id_gal[each[1]] += 1

        count_id_query = [0] *  id_label
        for each in query:
            count_id_query[each[1]] += 1

        return train, gallery, query
    
if __name__ == "__main__":
    s = sig()