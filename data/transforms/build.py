# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T
import random
import numpy as np
import cv2

from imgaug import augmenters as iaa
from PIL import Image
from torch.utils.data import Dataset

def change_sig_color(img, color=None):
    '''
    change image color from black white
    '''
    if color is None:
        color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]

    if isinstance(img, Image.Image):
        img = np.array(img)

    if len(img.shape) != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    sig_pos = np.where(img[:,:,0] != 255)
    img[:,:,0][sig_pos] = color[0]
    img[:,:,1][sig_pos] = color[1]
    img[:,:,2][sig_pos] = color[2]

    return Image.fromarray(img)


def custom_transform(img, prob=0.7):
    if random.uniform(0, 1) >= prob:
        return img

    # change img color
    img = change_sig_color(img)

    # rotate and add noise
    aug = iaa.Sequential([
        iaa.Affine(rotate=(-10,10)),
        iaa.AdditiveGaussianNoise(scale=(10,30))
    ], random_order=True)

    img_array = np.array(img)
    aug_img = aug(image=img_array)

    result = Image.fromarray(aug_img)
    
    return result


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    custom_tf = T.Lambda(custom_transform)
    if is_train:
        transform = T.Compose([
            custom_tf,
            T.ToTensor(),
            normalize_transform,
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            normalize_transform
        ])

    return transform


if __name__=="__main__":
    img = Image.open("/home/pdd/Desktop/workspace/sig_ReID/raw_sig/2/NFI-01601033.png")
    img = custom_transform(img, prob = 2)
    img.show()