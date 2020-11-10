# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
import math
import cv2
import numpy as np
from imgaug import augmenters as iaa
from PIL import Image
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            # img = Image.open(img_path).convert('RGB')
            cv2_img = cv2.imread(img_path, 0)
            _, cv2_img = cv2.threshold(cv2_img, 200, 255, cv2.THRESH_BINARY)
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
            PIL_img = Image.fromarray(cv2_img)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return PIL_img


class ImageDataset(Dataset):
    """ReID Dataset"""

    def __init__(self, cfg, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.size = cfg.INPUT.SIZE_TRAIN

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if len(self.dataset[index]) == 2:
            img_path, pid = self.dataset[index]
            img = read_image(img_path)
            img = self.resize_padding(img)

            if self.transform is not None:
                img = self.transform(img)
            return img, pid, img_path
        else:
            raise Exception("invalid dataset <Techainer>")
    
    def resize_padding(self, img):
        '''
        resize then pad image
        :params: img: PIL.Image
        '''
        if not isinstance(img, Image.Image):
            raise TypeError("Only support PIL Image")

        w, h = img.size
        new_h, new_w = cfg_size
        canvas = Image.new("RGB", (new_w, new_h), color=(255,255,255))

        if w >= h:
            resize_ratio = new_w / w
        else:
            resize_ratio = new_h / h

        img = img.resize((math.ceil(w*resize_ratio), 
                          math.ceil(h*resize_ratio)), Image.BICUBIC)

        current_w, current_h = img.size
        canvas.paste(img, (math.ceil((new_w-current_w)/2),
                           math.ceil((new_h-current_h)/2)))

        return canvas


if __name__=="__main__":
    img = read_image("/home/pdd/Desktop/workspace/sig_ReID/test.png")
    img.show()
