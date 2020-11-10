# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T
import random
import numpy as np


from imgaug import augmenters as iaa
from PIL import Image
from torch.utils.data import Dataset


def custom_transform(img, prob=0.5):
    if random.uniform(0, 1) >= prob:
        return img

    img_array = np.array(img)

    aug = iaa.Sequential([
        iaa.Affine(rotate=(-10,10)),
        iaa.AdditiveGaussianNoise(scale=(10,30))
    ], random_order=True)

    aug_img = aug(img_array)

    result = Image.fromarray(aug_img)
    
    return result


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    custom_tf = T.Lambda(custom_transform)
    if is_train:
        transform = T.Compose([
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
