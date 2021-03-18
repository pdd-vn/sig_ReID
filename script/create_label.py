import glob
import os
from tqdm import tqdm
import random

root_folder = "./data_collection/data_ky_lai"

label_path_image = []

list_root_folder = glob.glob(os.path.join(root_folder, "*"))

for index_sign, index_folder in tqdm(enumerate(list_root_folder)):
    current_folder = os.path.basename(os.path.normpath(index_folder))
    # id_folder = current_folder.split("_")[-1]
    id_folder = index_sign
    
    list_image = glob.glob(os.path.join(index_folder, "*", "*"))
    if len(list_image) == 0:
        print("hello")
    
    for path in list_image:
        type_sign = path.split("/")[-2]
        if type_sign == "Genuine":
            type_label = 1
        else:
            # type_label = 0
            continue

        basename = os.path.basename(path)
        # image_label = os.path.join(current_folder, basename)
        image_label = path.replace(root_folder,'.')
        # print(image_label)

        label_path_and_id = "{} {} {}\n".format(image_label, id_folder, type_label)
        label_path_image.append(label_path_and_id)

random.shuffle(label_path_image)
with open("./data_collection/label_signature.txt", "w") as f:
    f.writelines(label_path_image)