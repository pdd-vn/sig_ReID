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
import imgaug as ia
import PIL
from PIL import Image, ImageDraw, ImageFont
from . import utils
from torch.utils.data import Dataset
import glob
import random


corpus = [
    "Chữ ký",
    "Chữ ký thứ nhất",
    "Chữ ký thứ hai",
    "Mẫu chữ ký 1",
    "Mẫu chữ ký 2",
    "Mẫu chữ ký cũ đã dăng ký"
]

symbol_list = glob.glob("data/symbol/*.png")
stamp_list = glob.glob("data/stamp/*.png")
fonts = glob.glob("data/font/*")

aug = iaa.Sequential([
    # iaa.Sometimes(0.15, SpotLight()),
    # iaa.Sometimes(0.15, ParallelLight()),
    # iaa.Sometimes(0.15, LightFlare()),
    # Augment blur
    iaa.Sometimes(0.5,
        iaa.OneOf([
            iaa.AverageBlur(k=(3, 9)),
            iaa.GaussianBlur(sigma=(0., 1.5)),
            iaa.MedianBlur(k=(3, 7)),
            iaa.MotionBlur(k=(3, 9)),
            # iaa.imgcorruptlike.DefocusBlur(severity=(1, 3)),
            iaa.AveragePooling((1, 3)),
            # iaa.imgcorruptlike.GlassBlur(severity=(1, 2)),
            # iaa.UniformColorQuantization(n_colors=(4, 16)),
            # iaa.JpegCompression(compression=(40, 80)),
        ])
    ),
    # Condition
    iaa.Sometimes(0.25,
        iaa.OneOf([
            # iaa.Clouds(),
            iaa.MultiplyBrightness((0.5, 1.5)),
            iaa.CoarseDropout(0.02, size_percent=0.01, per_channel=1),
            iaa.PerspectiveTransform(scale=(0.01, 0.05), keep_size=False),
        ])
    ),
    iaa.Sometimes(0.2,
        iaa.ChangeColorTemperature((5000, 11000)),
        iaa.JpegCompression(compression=(50, 99))
    )
])


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def add_random_text(background):
    '''Random text in signature background
    '''
    bg_w, bg_h = background.size
    text = random.choice(corpus)
    draw = ImageDraw.Draw(background)
    font_path = random.choice(fonts)
    max_font_size = utils.get_fontsize(font_path, text, bg_w, bg_h//3)
    font_size = random.randint(int(max_font_size/3), max_font_size)
    font = ImageFont.truetype(font_path, font_size)
    x = random.randint(0, bg_w//2)
    y = random.randint(0, bg_h)
    draw.text((x,y), text, fill=(0,0,0), font=font)
    return background


def augment_imgaug(image):
    '''Augment image using imgaug
    Params
        :image: PIL.Image
    Return
        :image: PIL.Image
    '''
    image = np.array(image)
    image = aug.augment_image(image)
    image = Image.fromarray(image)
    return image


def augment_image(signature):
    ''' Augment signature 
    Params
        :image: PIL.Image
    Return
        :image: pIL.Image - augmented image
    '''
    
    aug_stt = False
    if random.random() < 0.4:
        signature = utils.blur_signature_std(signature, k_svd=(35, 65), color=[0, (0, 85), (130, 255)])
        aug_stt = True

    ratio_noise = random.random() 
    ratio_noise = 1
    wid_sig, hei_sig = signature.size
    wid_bg = wid_sig + int(random.uniform(0., 0.1)* wid_sig)
    hei_bg = hei_sig + int(random.uniform(0., 0.1)* hei_sig)
    background = Image.new('RGB', (wid_bg, hei_bg), color=(255,255,255))

    # Random add symbol
    if ratio_noise < 0.05: 
        noise_symbol = Image.open(random.choice(symbol_list)).convert("RGBA")
        color_noise = (random.randint(200, 255), random.randint(0, 100), random.randint(0, 50))
        background = utils.overlay_huge_transparent(background=background, foreground=noise_symbol, color=color_noise)
    # Random add stamp
    elif ratio_noise < 0.1:
        noise_stamp = Image.open(random.choice(stamp_list)).convert("RGBA")
        background = utils.overlay_huge_transparent(background=background, foreground=noise_stamp)    
    # Random add text
    elif ratio_noise < 0.15:
        background = add_random_text(background)

    # Random overlay signature on background
    coord_sig = (random.randint(0, wid_bg-wid_sig), random.randint(0, hei_bg-hei_sig))

    if not aug_stt:
        signature = utils.dilation_img(signature)
        signature = utils.create_transparent_image(signature)
        signature = utils.change_color_transparent(signature, color=((0, 30), (0,50), (10, 255)))
        background = augment_imgaug(background)

    background = utils.overlay_transparent(background=background, foreground=signature, coordinate=coord_sig, ratio=(0.5, 0.9))['filled_image']

    return background


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
            img = augment_image(img)
            # img = self.pre_processing(img)
            img = self.resize_padding(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, pid, img_path
        else:
            raise Exception("invalid dataset <Techainer>")


    def pre_processing(self, img):
        '''
        resize padding + binarizing image.
        '''
        img = self.resize_padding(img)
        # img_array = np.array(img)
        # _, img_array = cv2.threshold(img_array, 200, 255, cv2.THRESH_BINARY)
        # PIL_img = Image.fromarray(img_array)

        return PIL_img


    def resize_padding(self, img):
        '''
        resize then pad image
        :params: img: PIL.Image
        '''
        if not isinstance(img, Image.Image):
            raise TypeError("Only support PIL Image")

        w, h = img.size
        new_h, new_w = self.size
        canvas = Image.new("RGB", (new_w, new_h), color=(255,255,255))

        resize_ratio = min(new_w/w, new_h/h)

        img = img.resize((int(w*resize_ratio), 
                          int(h*resize_ratio)), Image.BICUBIC)

        current_w, current_h = img.size
        canvas.paste(img, (int((new_w-current_w)/2),
                           int((new_h-current_h)/2)))

        return canvas


if __name__=="__main__":
    # img = read_image("/home/pdd/Desktop/workspace/sig_ReID/test.png")
    search_path = "./raw_sig/*/*.png"
    import glob
    for idx, path in enumerate(glob.glob(search_path)):
        img = read_image(path)
        img = pre_processing(img)
        img.save("pre_processing/{}.png".format(idx))
    
