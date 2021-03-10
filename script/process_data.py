import cv2
import numpy as np

# Load image, convert to grayscale, Gaussian blur, Otsu's threshold
image = cv2.imread('/media/geneous/01D62877FB2A4900/Techainer/OCR/sig_ReID/data_collection/techainer/13_14_15_0_13_14_15_618_KAT/13_14_15_423 - 13_14_15_618/13_14_15_617/595dc28e168be5d5bc9a.jpg')
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (3,3), 0)
# thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1]

# Obtain bounding rectangle and extract ROI
x,y,w,h = cv2.boundingRect(thresh)
cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
ROI = original[y:y+h, x:x+w]

# # Add alpha channel
# b,g,r = cv2.split(ROI)
# alpha = np.ones(b.shape, dtype=b.dtype) * 50
# ROI = cv2.merge([b,g,r,alpha])

cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
cv2.imshow('thresh', thresh)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', image)
cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
cv2.imshow('ROI', ROI)
cv2.waitKey()

def crop_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    x,y,w,h = cv2.boundingRect(thresh)
    ROI = original[y:y+h, x:x+w]

    return ROI
