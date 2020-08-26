import numpy as np
from PIL import Image
from mlchain.client import Client

model = Client(api_address='http://localhost:5000').model()
img1 = r"/home/pdd/Desktop/sandbox/reid-strong-baseline/f1/MicrosoftTeams-image.png"
img2 = r"/home/pdd/Desktop/sandbox/reid-strong-baseline/f2/MicrosoftTeams-image (11).png"
result = model.verify(img1, img2, threshold = 0.2)
print(result)  

# model = Client(api_address='http://localhost:5000').model()
# img1 = Image.open(r"/home/pdd/Desktop/sandbox/reid-strong-baseline/f1/MicrosoftTeams-image.png")
# img2 = Image.open(r"/home/pdd/Desktop/sandbox/reid-strong-baseline/f2/MicrosoftTeams-image (11).png")
# result = model.verify2(img1, img2, threshold=0.2)
# print(result)  