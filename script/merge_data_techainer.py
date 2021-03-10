import os
import glob
import cv2
import numpy as np 
import shutil

from tqdm import tqdm

techainer_path = "/media/geneous/01D62877FB2A4900/Techainer/OCR/sig_ReID/data_collection/techainer/all_data_techainer"
seabank_path = "/media/geneous/01D62877FB2A4900/Techainer/OCR/data/Seabank/data_chu_ky/labeled/cropped_sign_image"

t_folders = glob.glob(os.path.join(techainer_path, "*"))
s_folders = glob.glob(os.path.join(seabank_path, "*"))

output_dataset = "data_collection/prepared_dataset"
os.makedirs(output_dataset, exist_ok=True)

for index, folder_path_tech in tqdm(enumerate(t_folders), total=len(t_folders)):
    base_id = os.path.basename(folder_path_tech)

    folder_path_sb = os.path.join(seabank_path, base_id)
    if folder_path_sb in s_folders:
        t_imgs = glob.glob(os.path.join(folder_path_tech, "*"))
        s_imgs = glob.glob(os.path.join(folder_path_sb, "*"))
        folder_id_full = os.path.join(output_dataset,f"{index}_{base_id}_full")
        folder_id_sign = os.path.join(output_dataset,f"{index}_{base_id}_signature")

        for path in t_imgs:
            img = cv2.imread(path)
            
            basename = os.path.basename(path)
            img_name = "{}.jpg".format(basename.split(".")[0]).strip()
            if basename.startswith("Same_signature") or basename.startswith(" Same_signature"):
                new_folder = os.path.join(folder_id_sign, "Genuine")
                os.makedirs(new_folder, exist_ok=True)

            elif basename.startswith("Forgery_signature") or basename.startswith(" Forgery_signature"):
                new_folder = os.path.join(folder_id_sign, "Forgery")
                os.makedirs(new_folder, exist_ok=True)
            
            elif basename.startswith("Same_full") or basename.startswith(" Same_full"):
                new_folder = os.path.join(folder_id_full, "Genuine")
                os.makedirs(new_folder, exist_ok=True)

            elif basename.startswith("Forgery_full") or basename.startswith(" Forgery_full"):
                new_folder = os.path.join(folder_id_full, "Forgery")
                os.makedirs(new_folder, exist_ok=True)

            else:
                print("path {} is not right type in tech".format(img_name))
                continue

            new_path = os.path.join(new_folder, img_name)
            cv2.imwrite(new_path, img)


        for path in s_imgs:
            # if basename.startswith("signature"):
            #     img = cv2.imread(path)
            img = cv2.imread(path)
            
            basename = os.path.basename(path)
            img_name = "{}.jpg".format(basename.split(".")[0])
            if basename.startswith("signature"):
                new_folder = os.path.join(folder_id_sign, "Genuine")
                os.makedirs(new_folder, exist_ok=True)

            
            elif basename.startswith("full"):
                new_folder = os.path.join(folder_id_full, "Genuine")
                os.makedirs(new_folder, exist_ok=True)

            else:
                print("path {} is not right type in sb".format(img_name))
                continue

            new_path = os.path.join(new_folder, img_name)
            cv2.imwrite(new_path, img)