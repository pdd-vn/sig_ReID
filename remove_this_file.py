import glob
import cv2
import numpy as np
import os


root_path = 'data_collection/Seabank'
list_data = glob.glob(os.path.join(root_path, '*', "*", "*"))
for id_path in list_data:
    list_id = glob.glob(os.path.join(id_path, ".DS_Store"))
    for path in list_id:
        os.remove(path)
