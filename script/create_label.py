import glob
import os
from tqdm import tqdm
import random
import natsort

root_folder = "./dataset"

label_path_image = []

list_root_folder = glob.glob(os.path.join(root_folder, "*"))
id_folder = 0
for index_sign_old, index_folder in tqdm(enumerate(list_root_folder)):
    current_folder = os.path.basename(os.path.normpath(index_folder))
    # id_folder = current_folder.split("_")[-1]
    sub_folders = glob.glob(os.path.join(index_folder, "*"))

    for sub_path in sub_folders:
        list_image = glob.glob(os.path.join(sub_path, "*"))
        if len(list_image) == 0:
            print(index_folder)
            continue
            
        
        for path in list_image:
            type_label = 1

            basename = os.path.basename(path)
            # image_label = os.path.join(current_folder, basename)
            image_label = path.replace(root_folder,'.')
            # print(image_label)

            label_path_and_id = "{} {} {}\n".format(image_label, id_folder, type_label)
            label_path_image.append(label_path_and_id)
    
        id_folder += 1

random.shuffle(label_path_image)
with open("./label_signature.txt", "w") as f:
    f.writelines(label_path_image)