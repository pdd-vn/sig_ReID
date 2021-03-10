import glob
import os
from tqdm import tqdm

root_path = "/media/geneous/01D62877FB2A4900/Techainer/OCR/sig_ReID/data_collection/techainer/all_data_techainer"

folders = glob.glob(os.path.join(root_path, "*"))

list_rm = []
list_overlap = []

for folder_path in folders:
    images = glob.glob(os.path.join(folder_path, "*"))
    valid = False
    for path in images:
        image_name = os.path.basename(path)
        # if image_name.startswith("Forgery") or image_name.startswith("Same"):
        #     valid=True
        #     break
        if image_name.startswith("signature") or image_name.startswith("full"):
            list_overlap.append(path)
    # if not valid:
    #     list_rm.append(folder_path)

for path in list_overlap:
    print(path)
    os.remove(path)


