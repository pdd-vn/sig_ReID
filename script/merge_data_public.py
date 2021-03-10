import os
import glob
import shutil
import cv2
import numpy as np
from tqdm import tqdm

output_folder = "data_collection/prepared_public_data"
os.makedirs(output_folder, exist_ok=True)

root_folder = "/media/geneous/01D62877FB2A4900/Techainer/OCR/sig_ReID/data_collection/public_dataset/processed_pucblic_dataset"

list_folders = glob.glob(os.path.join(root_folder, "*"))

for sub_folder in list_folders:
    sub_folder_name = os.path.basename(sub_folder)
    list_ids = glob.glob(os.path.join(sub_folder, "*"))
    for idx in tqdm(list_ids):
        base_id = os.path.basename(idx)
        new_base_id = "{}_{}".format(sub_folder_name, base_id)
        new_id_path = os.path.join(output_folder, new_base_id)
        
        new_gen_folder = os.path.join(new_id_path, "Genuine")
        os.makedirs(new_gen_folder, exist_ok=True)

        new_forg_folder = os.path.join(new_id_path, "Forgery")
        os.makedirs(new_forg_folder, exist_ok=True)

        gen_id_path = os.path.join(idx, "Genuine")
        if os.path.isdir(gen_id_path):
            list_gen = glob.glob(os.path.join(gen_id_path, "*"))
            for path in list_gen:
                img = cv2.imread(path)
                basename = os.path.basename(path)
                new_path = os.path.join(new_gen_folder, basename)
                cv2.imwrite(new_path, img)

        
        else:
            gen_id_path = os.path.join(idx, "genuine")
            if os.path.isdir(gen_id_path):
                list_gen = glob.glob(os.path.join(gen_id_path, "*"))
                for path in list_gen:
                    img = cv2.imread(path)
                    basename = os.path.basename(path)
                    new_path = os.path.join(new_gen_folder, basename)
                    cv2.imwrite(new_path, img)
            
        
        
        forg_id_path = os.path.join(idx, "Forgery")
        if os.path.isdir(forg_id_path):
            list_gen = glob.glob(os.path.join(forg_id_path, "*"))
            for path in list_gen:
                img = cv2.imread(path)
                basename = os.path.basename(path)
                new_path = os.path.join(new_forg_folder, basename)
                cv2.imwrite(new_path, img)

        
        else:
            forg_id_path = os.path.join(idx, "forgery")
            if os.path.isdir(gen_id_path):
                list_gen = glob.glob(os.path.join(forg_id_path, "*"))
                for path in list_gen:
                    img = cv2.imread(path)
                    basename = os.path.basename(path)
                    new_path = os.path.join(new_forg_folder, basename)
                    cv2.imwrite(new_path, img)
            