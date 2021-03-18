import os
import glob
import shutil
from tqdm import tqdm

output_folder_tp = 'data_collection/TP/processed_data_tp'
os.makedirs(output_folder_tp, exist_ok=True)

root_path = 'data_collection/TP/crop_tp'
all_id = glob.glob(os.path.join(root_path, "*"))

for each_id in tqdm(all_id):
    
    list_sub_type = glob.glob(os.path.join(each_id, "*"))
    base_name = os.path.basename(each_id)
    for sub_type in list_sub_type:
        base_sub_type = os.path.basename(sub_type)
        new_folder_name = "{}_{}".format(base_name, base_sub_type)
        new_folder_path = os.path.join(output_folder_tp, new_folder_name)
        print(sub_type)
        print(new_folder_path)
        shutil.copytree(sub_type, new_folder_path)