import glob
import os
from tqdm import tqdm
import random
import shutil

root_folder = "./dataset"

label_path_image = []

list_root_folder = glob.glob(os.path.join(root_folder, "*"))

for index_sign, index_folder in tqdm(enumerate(list_root_folder)):
    current_folder = os.path.basename(os.path.normpath(index_folder))

    id_folder = index_sign
    
    list_image = glob.glob(os.path.join(index_folder, "*.jpg")) + glob.glob(os.path.join(index_folder, "*.png"))
    
    if len(list_image) == 0:
        continue
    
    new_folder_name = "Genuine"
    new_folder_path = os.path.join(index_folder, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)

    print(len(list_image))
    for path in list_image:
        basename = os.path.basename(path)
        new_path = os.path.join(new_folder_path, basename)
        shutil.move(path, new_path)
    
#     for path in list_image:
#         type_sign = path.split("/")[-2]
#         if type_sign == "Forgery":
#             type_label = 0
#         else:
#             type_label = 1

#         basename = os.path.basename(path)
#         # image_label = os.path.join(current_folder, basename)
#         image_label = path.replace(root_folder,'.')
#         # print(image_label)

#         label_path_and_id = "{} {} {}\n".format(image_label, id_folder, type_label)
#         label_path_image.append(label_path_and_id)

# random.shuffle(label_path_image)
# with open("./label_signature.txt", "w") as f:
#     f.writelines(label_path_image)