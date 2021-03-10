import os
from tqdm import tqdm
import numpy as np
import cv2
import glob

def crop_object(image):
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    x,y,w,h = cv2.boundingRect(thresh)
    ROI = original[y:y+h, x:x+w]

    return ROI

root_path = "/media/geneous/01D62877FB2A4900/Techainer/OCR/sig_ReID/data_collection/techainer/all_data_techainer"

folders = glob.glob(os.path.join(root_path, "*"))

for folder_path in tqdm(folders):
    images = glob.glob(os.path.join(folder_path, "*"))
    for path in images:
        image_name = os.path.basename(path)

        img = cv2.imread(path)
        cropped_img = crop_object(img)
        try:

            cv2.imwrite(path, cropped_img)
        except:
            import ipdb; ipdb.set_trace()

