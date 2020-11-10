import numpy as np
import torch
import os
import torchvision.transforms as T
from PIL import Image
from mlchain.base import ServeModel
from scipy.spatial import distance

class Sig_Ver_Model():
    def __init__(self, model_path):
        with torch.no_grad():
            self.model = torch.load(model_path, map_location='cpu')
            self.model.eval()
    

    def build_transforms(self):
        normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = T.Compose([
                T.Resize([125, 250]),
                T.ToTensor(),
                normalize_transform
            ])

        return transform


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
        # import ipdb; ipdb.set_trace()
        img1 = transform(img1).unsqueeze(0)
        img2 = transform(img2).unsqueeze(0)
        
        feat1 = self.model(img1).detach().numpy()
        feat2 = self.model(img2).detach().numpy()

        dis = distance.cosine(feat1, feat2)
        dis = distance.euclidean(feat1, feat2)
        if dis < threshold:
            return True, dis
        else:
            return False, dis
    

    def verify2(self, img1, img2, threshold=0.4):  
        '''
        verify 2 signature using image file.
        input: - img1, img2: image to verify.
               - threshold: threshold for cosine similarity evaluation.
        output: True/False
        '''
        transform = self.build_transforms()
        img1 = transform(img1).unsqueeze(0)
        img2 = transform(img2).unsqueeze(0)
        
        feat1 = self.model(img1).detach().numpy()
        feat2 = self.model(img2).detach().numpy()

        dis = distance.cosine(feat1, feat2)
        dis = distance.euclidean(feat1, feat2)
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
        f1_list = [img for img in os.listdir(folder1_path)]
        f2_list = [img for img in os.listdir(folder2_path)]
        
        for img1 in f1_list:
            for img2 in f2_list:
                img1_path = os.path.join(folder1_path, img1)
                img2_path = os.path.join(folder2_path, img2)

                verified, dis = self.verify(img1_path, img2_path)
                if verified:
                    print("{} = {} - dis: {}".format(img1, img2, dis))
                else:
                    print("{} != {} - dis: {}".format(img1, img2, dis))

# def main():
#     model = Sig_Ver_Model()
#     model.multiple_pair_verify("./f1", "./f2")


# model = Sig_Ver_Model()
# serve_model = ServeModel(model)

if __name__=="__main__":
    # #main()
    # from mlchain.server import FlaskServer
    # # Run flask model with upto 12 threads
    # FlaskServer(serve_model).run(port=5000, threads=12)

    model = Sig_Ver_Model(model_path="test.pth")
    print(model.verify("/home/pdd/Desktop/workspace/sig_ReID/sig_result/60.png", "/home/pdd/Desktop/workspace/sig_ReID/sig_result/61.png"))