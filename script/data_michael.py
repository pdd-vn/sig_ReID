import glob
import os
from tqdm import tqdm

root_path = "/media/geneous/01D62877FB2A4900/Techainer/OCR/sig_ReID/data_collection/techainer/3_4_5_159_3_4_5_315_michael"

images = glob.glob(os.path.join(root_path, "*", "*"))

for path in tqdm(images):
    if path.endswith("jpg"):
        new_path = path.replace("jpg", "png")
        os.rename(path, new_path)
