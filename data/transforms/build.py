# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T
import random
import numpy as np

from .transforms import RandomErasing
from imgaug import augmenters as iaa
from PIL import Image
from torch.utils.data import Dataset

def augment(img):
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-10,10)),
        iaa.AdditiveGaussianNoise(scale=(10,30))
    ], random_order=True)
    return seq(image=img)


def resize_padding(cfg_size):
    def transform(img, size=cfg_size):
        w, h = img.size
        new_h, new_w = cfg_size

        ratio = new_h / h 
        resized_w = int(w*ratio)
        if resized_w%2 != 0:
            resized_w += -1
        if resized_w >= new_w:
            return T.Resize([new_h, new_w])(img)
        else:
            pad_left_right = int((new_w - resized_w)/2)
            resized_img = T.Resize([new_h, resized_w])(img)
            return T.Pad((pad_left_right,0),fill=(255,255,255))(resized_img)
    return transform
        

def custom_transform(img, prob=0.5):
    if random.uniform(0, 1) >= prob:
        return img

    img_array = np.array(img)
    aug_img = augment(img_array)
    result = Image.fromarray(aug_img)
    return result


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    custom_tf = T.Lambda(custom_transform)
    res_pad = T.Lambda(resize_padding(cfg.INPUT.SIZE_TRAIN))
    if is_train:
        transform = T.Compose([
            res_pad,
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            custom_tf,
            T.ToTensor(),
            normalize_transform,
        ])
    else:
        transform = T.Compose([
            res_pad,
            T.ToTensor(),
            normalize_transform
        ])

    return transform
