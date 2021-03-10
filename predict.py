import numpy as np
import torch
import os
import torchvision.transforms as T
from PIL import Image
from mlchain.base import ServeModel
from scipy.spatial import distance
import cv2
import glob

class Sig_Ver_Model():
    def __init__(self, model_path):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

    def build_transforms(self):
        normalize_transform = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform = T.Compose([
                T.ToTensor(),
                normalize_transform
            ])
        return transform


    def resize_padding(self, img, size=(250, 125)):
        '''
        resize then pad image
        :params: img: PIL.Image
        '''
        if not isinstance(img, Image.Image):
            raise TypeError("Only support PIL Image")

        w, h = img.size
        new_w, new_h = size
        canvas = Image.new("RGB", (new_w, new_h), color=(255,255,255))

        resize_ratio = min(new_w/w, new_h/h)

        img = img.resize((int(w*resize_ratio),
                            int(h*resize_ratio)), Image.BICUBIC)

        current_w, current_h = img.size
        canvas.paste(img, (int((new_w-current_w)/2),
                            int((new_h-current_h)/2)))

        return canvas


    def verify(self, img1_path, img2_path, threshold=0.4):   
        '''
        verify 2 signature using image path.
        input: - img1_path, img2_path: path to image to verify.
               - threshold: threshold for cosine similarity evaluation.
        output: True/False
        '''
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        transform = self.build_transforms()

        img1 = self.resize_padding(img1)
        img2 = self.resize_padding(img2)
        
        img1_cv = np.array(img1)
        img2_cv = np.array(img2)

        cv2.imshow("1", img1_cv[:,:,::-1])
        cv2.imshow("2", img2_cv[:,:,::-1])

        img1 = transform(img1).unsqueeze(0).to(self.device)
        img2 = transform(img2).unsqueeze(0).to(self.device)
        
        feat1 = self.model(img1).detach().numpy()
        feat2 = self.model(img2).detach().numpy()

        dis = distance.cosine(feat1, feat2)
        similarity = 1 - dis
        print("Similarity: ", similarity)
        cv2.waitKey(0)

        if dis < threshold:
            return True, dis
        else:
            return False, dis


    def multiple_pair_verify(self, folder1_path, folder2_path):
        '''
        verify signatures in 2 folders pairwises.
        input: - folder1_path, folder2_path: path to folders to verify.
               - threshold: threshold for cosine similarity evaluation.
        '''
        f1_list = glob.glob(os.path.join(folder1_path, "*"))
        f2_list = glob.glob(os.path.join(folder2_path, "*"))
        
        for img1_path in f1_list:
            print("________________________________")
            for img2_path in f2_list:
                basename1 = os.path.basename(img1_path)
                basename2 = os.path.basename(img2_path)
                print("{}_+++_{}".format(basename1, basename2))
                
                verified, dis = self.verify(img1_path, img2_path)
                # if verified:
                #     print("{} = {} - dis: {}".format(img1, img2, dis))
                # else:
                #     print("{} != {} - dis: {}".format(img1, img2, dis))


if __name__=="__main__":
    model = Sig_Ver_Model(model_path="weights/model_resnet_ibn_19012021_ep60_circle.pth")

    path_1 = "/media/geneous/01D62877FB2A4900/Techainer/OCR/data/Seabank/data_chu_ky/labeled/crop_dataset_po/crop_dataset/3_4_5_1.jpg"
    path_2 = "/media/geneous/01D62877FB2A4900/Techainer/OCR/data/Seabank/data_chu_ky/labeled/crop_dataset_po/crop_dataset/3_4_5_1.jpg"
    model.multiple_pair_verify(path_1, path_2)