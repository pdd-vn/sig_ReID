# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
import cv2
import numpy as np
# from imgaug import augmenters as iaa
from PIL import Image
from torch.utils.data import Dataset

# def augment(img):
#     seq = iaa.Sequential([
#         iaa.Affine(rotate=(-15,15)),
#         iaa.AdditiveGaussianNoise(scale=(10,30))
#     ], random_order=True)
#     return seq(image=img)
def pre_processing(img, pf_shape=[500,1000]):
    pil_image = img.convert('RGB') 
    open_cv_img = np.array(pil_image) 
    # Convert RGB to BGR 
    open_cv_img = open_cv_img[:, :, ::-1].copy() 

    h, w, _ = open_cv_img.shape
    pf_h, pf_w = pf_shape
    
    ratio = max(np.ceil(pf_h/h),np.ceil(pf_w/w))
    new_h = h*ratio
    new_w = w*ratio
    
    pad_h = np.floor((new_h-h)/2).astype(np.uint8)
    pad_w = np.floor((new_w-w)/2).astype(np.uint8)
    color = [255,255,255] #wait for genos
    img = cv2.copyMakeBorder(open_cv_img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[255,255,255])
    img = cv2.resize(img, (pf_w, pf_h))
    img = Image.fromarray(img)
    return img

def read_image(img_path):
    
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            # if not osp.exists(img_path) and img_path.endswith("aug"):
            #     img_path = img_path[:-4]
            #     img = Image.open(img_path).convert('RGB')
            #     img_aug = augment(img)
            #     img = img_aug
            #     got_img = True
            # elif osp.exists(img_path):
            #     img = Image.open(img_path).convert('RGB')
            #     got_img = True
            # else:
            #     raise IOError("{} does not exist".format(img_path))
            img = Image.open(img_path).convert('RGB')
            img = pre_processing(img)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img
    

class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if len(self.dataset[index]) == 3:
            img_path, pid, camid = self.dataset[index]
            img = read_image(img_path)

            if self.transform is not None:
                img = self.transform(img)
            return img, pid, camid, img_path
        elif len(self.dataset[index]) == 2:
            img_path, pid = self.dataset[index]
            img = read_image(img_path)

            if self.transform is not None:
                img = self.transform(img)
            return img, pid, img_path
        else:
            raise Exception("invalid dataset <made by pdd>")
