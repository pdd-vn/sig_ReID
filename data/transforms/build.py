# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing
from PIL import PIL_image
from imgaug import augmenters as iaa
from PIL import Image
from torch.utils.data import Dataset

def augment(img):
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-10,10)),
        iaa.AdditiveGaussianNoise(scale=(10,30))
    ], random_order=True)
    return seq(image=img)

def custom_transform(img, prob=0.5):
    if random.uniform(0, 1) >= prob:
        return img

    img_array = np.array(img)
    aug_img = augment(img_array)
    result = Image.fromarray(aug_img)
    return result

def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    custom_tf = T.lambda(custom_transform)
    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            custom_tf,
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
