import numpy as np 
import os 
import cv2 
from PIL import Image, ImageFont, ImageFont
import PIL
import random
import imgaug.augmenters as iaa
from copy import deepcopy



def tlwh_to_polygon(left, top, width, height):
    '''Convert bounding box (left, top, width, height) to polygon 4 point
    '''
    x, y = left, top
    return [
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height)
    ]


def get_fontsize(font_path, text, max_width, max_height):
    ''' Get fontsize fit bounding box - PIL
    '''
    fontsize = 1
    font = ImageFont.truetype(font_path, fontsize) 
    height_text = font.getsize(text)[1] #- font.getoffset(text)[1]
    width_text = font.getsize(text)[0]  #- font.getoffset(text)[0]

    while True:
        font = ImageFont.truetype(font_path, fontsize)
        height_text = font.getsize(text)[1] #- font.getoffset(text)[1]
        width_text = font.getsize(text)[0] #- font.getoffset(text)[0]
        if height_text > max_height or width_text > max_width:
            fontsize -= 1
            font = ImageFont.truetype(font_path, fontsize)
            height_text = font.getsize(text)[1] #- font.getoffset(text)[1]
            width_text = font.getsize(text)[0]  #- font.getoffset(text)[0]
            if height_text > max_height or width_text > max_width:
                fontsize -= 1
                break
            else:
                break
        fontsize += 2
    return fontsize


def get_coord_text(font, text, xy):
    ''' Get bounding box fit mast text - PIL
    Return
    ------
    :Bounding box: (left, top, width, height)
    '''
    first=True
    left_text = None
    top_text = None
    right_text = None
    bot_text = None

    for k, char in enumerate(text):
        if char == " ":
            continue
        # Get coordinate of each letters
        bottom_1 = font.getsize(text[k])[1]
        right, bottom_2 = font.getsize(text[:k+1])
        bottom = bottom_1 if bottom_1<bottom_2 else bottom_2
        width_char, height_char = font.getmask(char).size
        right += xy[0]
        bottom += xy[1]
        top = bottom - height_char
        left = right - width_char
        # Get coordinate of the first letter
        if first:
            left_text = left
            top_text = top
            first = False
            bot_text = bottom
            

        right_text = right
        bot_text = max(bot_text, bottom)
        top_text = min(top_text, top)

    if None in [left_text, right_text, bot_text, top_text]:
        raise ValueError("Invalid coordinate of the text. \
                        Expect 4 numbers but get None!")
        # return (xy[0], xy[1], 0, 0)
    width_text = right_text - left_text
    height_text = bot_text - top_text

    return (left_text, top_text, width_text, height_text)


def overlay_huge_transparent(background:PIL.Image, foreground:PIL.Image, color=None, ratio=None):
    ''' Overlay huge transparent image on background
    Params
    ------
    :background: PIL.Image 
    :foreground: PIL.Image - RGBA image

    Returns
    -------
    :image: PIL.Image
    '''

    bg_w, bg_h = background.size

    cur_fore_h, cur_fore_w = foreground.size
    if random.uniform(0, 1) < 0.5:  # big
        ratio_size = random.uniform(3, 4)
    else:
        ratio_size = random.uniform(0.9, 2)

    new_fore_h = int(ratio_size * bg_h)
    new_fore_w = int((new_fore_h/cur_fore_h) * cur_fore_w)
    foreground = foreground.resize((new_fore_w, new_fore_h))    

    x = random.randint(int(bg_w/2 - new_fore_w), int(bg_w/2))
    y = random.randint(int(bg_h/2 - new_fore_h), int(bg_h/2))
    
    background = overlay_transparent(background=background, foreground=foreground,  \
                                    coordinate=(x,y), color=color, ratio=ratio)['filled_image']

    return background

    
def overlay_transparent(background:PIL.Image, foreground:PIL.Image, coordinate=None, remove_bg=False, color=None, ratio=None):
    ''' Overlay transparent image on background
    Params
    ------
    :background: PIL.Image 
    :foreground: PIL.Image - RGBA image
    :coordinate: list(x,y) - Left-top coordinate on background
    :remove_bg: bool - remove background of inserted image
    :ratio: ratio of opacity foreground

    Returns
    -------
    :image: PIL.Image
    '''
    org_background = background.copy()
    if org_background.mode == "P":
        org_background = org_background.conver("RGB")

    bg_w, bg_h = background.size 
    fg_w, fg_h = foreground.size
    # Left-top coordinate of inserted image
    if coordinate is None:
        x = random.randint(0, bg_w - fg_w)
        y = random.randint(0, bg_h - fg_h)
    else:
        x, y = coordinate

    wid = fg_w
    hei = fg_h
    
    # If remove background if inserted image
    if remove_bg:
        inserted_area = np.array(background)[y:y+hei, x:x+wid, :]
        dominant_color = find_dominant_color(inserted_area)
        background = np.array(background)
        background[y:y+hei, x:x+wid, :] = dominant_color
        background = Image.fromarray(background)
    
    # Change color of foreground
    if color is not None:
        foreground = change_color_transparent(foreground, color)
    
    if foreground.mode != "RGBA":
        # print("Bad transparent foreground when overlay into the background.")
        foreground = foreground.convert("RGBA")
        
    # Overlay transparent
    background.paste(foreground, (x,y), foreground)
    if ratio is None:
        ratio = 1
    elif isinstance(ratio, tuple):
        ratio = random.uniform(ratio[0], ratio[1])
    elif isinstance(ratio, float):
        pass
    else:
        raise ValueError("ratio blend must be None or float or tuple, get type {}".format(type(ratio)))

    background = Image.blend(org_background, background, ratio)

    return {
        "filled_image":background,
        "bbox": (x,y, wid, hei)
    }


def change_color_transparent(image, color=(None, None, None), smooth=False):
    if isinstance(image, PIL.Image.Image):
        if image.mode == "L":
            image = image.convert("RGB")
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else: # len(image.shape) == 3
            pass

    for i in range(3):
        if color[i] == "random":
            if smooth:
                color_channel = random.randint(0, 255)
                image[:,:,i] = np.maximum(np.abs(image[:,:,i] - color_channel), image[:,:,i])
            else:
                image[:,:,i] = random.randint(0, 255)
        elif isinstance(color[i], tuple):
            if smooth:
                color_channel = random.randint(color[i][0], color[i][1])
                image[:,:,i] = np.maximum(np.abs(image[:,:,i] - color_channel), image[:,:,i])
            else:
                image[:,:,i] = random.randint(color[i][0], color[i][1])
        elif isinstance(color[i], int):
            if smooth:
                image[:,:,i] = np.maximum(np.abs(color[i] - image[:,:,i]), image[:,:,i])
            else:
                image[:,:,i] = color[i]

    image = Image.fromarray(image)
    return image


def create_transparent_image(image, threshold=225):
    '''Create transparent image from white background image
    Params
    ------
    :image 
    :threshold: threshold to remove white background

    Return
    ------
    image: PIL.Image
    '''
    if isinstance(image, str):
        image = Image.open(image).convert("RGBA")
    else: # PIL.Image
        image = image.convert("RGBA")
    
    # # Version PIL
    # width, height = image.size
    # pixel_data = image.load()
    # # Set alpha channel to zerp pixel value > threshold
    # for y in range(height):
    #     for x in range(width):
    #         if all(np.asarray(pixel_data[x,y][:3]) > threshold):
    #             pixel_data[x, y] = (255, 255, 255, 0)
    #         else:
    #             pixel_data[x, y] = (0,0,0,255)

    image_rgba = np.array(image)
    image_rgb = deepcopy(image_rgba[:,:,:3])
    _, bin_img = cv2.threshold(image_rgb, threshold, 255, cv2.THRESH_BINARY)
    height, width = image_rgba.shape[:2]
    # Transparent mask
    # transparent_area = np.any(image_rgb > [threshold, threshold, threshold], axis=-1)
    transparent_area = np.any(bin_img > threshold, axis=-1)
    image_rgba[transparent_area, -1] = 0 

    image = Image.fromarray(image_rgba)
    return image


def find_dominant_color(background: np.ndarray):
    '''find the dominant color on background
    
    Params
    ------
    :background: PIL.Image - "RGB"
    '''
    image = Image.fromarray(background)
    #Resizing parameters
    width, height = 448, 448
    # image = Image.fromarray(background)
    image = image.resize((width, height),resample = 0)
    #Get colors from image object
    pixels = image.getcolors(width * height)
    #Sort them by count number(first element of tuple)
    sorted_pixels = sorted(pixels, key=lambda t: t[0])
    #Get the most frequent color
    dominant_color = sorted_pixels[-1][1]
    return dominant_color


def tlwh_2_yolo_format(bbox, bg_shape):
    x, y, w, h = bbox
    bg_w, bg_h = bg_shape
    x_center = np.round(((x+w/2) / bg_w), 6)
    y_center = np.round(((y+h/2) / bg_h), 6)
    width = np.round((w/bg_w), 6)
    height = np.round((h/bg_h), 6)
    return (x_center, y_center, width, height)


def gen(ind):
    path = data[ind]
    name_img = os.path.basename(path)
    if name_img.endswith(".jpg"):
        name_img = "{}.png".format(name_img.split(".")[0])
    output_folder = os.path.join("data/transparent_signature")
    os.makedirs(output_folder, exist_ok=True)

    image = cv2.imread(path, 0)
    _, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    image = Image.fromarray(image)
    image = image.convert("RGB")
    # image = Image.open(path)
    wid, hei = image.size 
    ratio = hei/wid
    # image = image.resize((new_wid, new_hei))
    image = create_transparent_image(image, threshold=225)
    out_image_path = os.path.join(output_folder, name_img)
    image.save(out_image_path)

# Update 10/11/2020
def svd_compress(img, k):
    '''
    singular value decomposition
    input: - img: your image in gray scale
            - k: number of kept eigenvalues
    output: truncated image.
    '''
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    else:
        pass
    if isinstance(k, tuple):
        k = random.randint(k[0], k[1])
    elif isinstance(k, int):
        pass
    else:
        raise ValueError("Do not support type {} of k".format(type(k)))

    u, s, v = np.linalg.svd(img)
    try:
        compressed_img = np.dot(u[:,:k],np.dot(np.diag(s[:k]),v[:k,:]))
        compressed_img = Image.fromarray(compressed_img)
    except ValueError:
        print("Error SVD")
        return Image.fromarray(img)
    return compressed_img


def toNumpy(img):
    return np.array(img)


def erosion_img(img, k_size=3):
    '''
    img: PIL.Image
    '''
    if isinstance(img, PIL.Image.Image):
        img = img.convert("L")
        img = np.array(img)
    else:
        pass
    kernel = np.ones((k_size, k_size), dtype=np.uint8)
    img = cv2.erode(img, kernel)
    img = Image.fromarray(img)
    return img


def dilation_img(img, k_size=None):
    '''
    img: PIL.Image
    '''
    if isinstance(img, PIL.Image.Image):
        img = img.convert("L")
        img = np.array(img)
    else:
        pass
    if k_size is None:
        k_size = random.randint(1, 3)
        # k_size = 2
    kernel = np.ones((k_size, k_size), dtype=np.uint8)
    img = cv2.dilate(img, kernel)
    img = Image.fromarray(img)
    return img


def blur_img(img, k=None):
    '''blur image using imgaug
    img: PIL.Image
    '''
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    else:
        pass

    if k is None:
        k = (3, 5)
    elif not (isinstance(k, tuple) or isinstance(k, int)):
        raise TypeError("Does not support k type: {}".format(type(k)))

    aug_blur = iaa.AverageBlur(k=k)
    img_blur = aug_blur.augment_image(img)
    img_blur = Image.fromarray(img_blur)
    return img_blur


def binarize_img(img, threshold=200):
    if isinstance(img, PIL.Image.Image):
        img = img.convert("L")
        img = np.array(img)
    else:
        if len(img.shape) == 3:
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            else:
                raise ValueError("Number channel of image must have ")
    
    assert len(img.shape) == 2, "Must be grayscale image!"
    _, bin_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    bin_img = Image.fromarray(bin_img)
    return bin_img


def blur_signature_std(img, k_blur=None, k_svd=None, color=[None, None, None]):
    ''' Blur signature
    Params
        :img: PIL.Image
        :k_blur: intensity blur
        :k_svd: intensity compress
    '''
    # Create blur shadow signature
    # color_shadow=[None, None, None]
    color_shadow=[None, None, 255]

    color = list(color)
    for i in range(len(color)):
        if color[i] is None:
            color[i] = random.randint(0, 255)
    if random.random() < 0:
        for i in range(len(color)):
            if isinstance(color[i], int):
                color_shadow[i] = random.randint(int(color[i]*1.2), int(color[i]*2))
            elif isinstance(color[i], tuple):
                color_shadow[i] = random.randint(int(color[i][0]), int(color[i][1]))
    # Binarize image (low=0, high=255)
    img = binarize_img(img)
    
    # Erode image
    aug_erode = False
    aug_dilate = True
    if random.random() < 0.1:
        aug_erode = True
    if random.random() < 0.1:
        aug_dilate = False

    # if aug_erode:
    #     k_erode = random.randint(2, 3)
    #     img = erosion_img(img, k_erode)

    # Blur eroded image
    if k_blur is None:
        k_blur = random.randint(2, 3)
    shadow_blur = blur_img(img, k=k_blur)

    # Compress blur image
    if k_svd is None:
        k_svd = random.randint(25, 55)
    k_svd = 10
    shadow = svd_compress(shadow_blur, k=k_svd)

    # Augment and transparent image
    shadow_trans = create_transparent_image(shadow)
    shadow_trans = change_color_transparent(shadow_trans, color=color_shadow, smooth=True)
    # shadow_trans.show()

    # # Dilate raw image
    if aug_dilate:
        k_dilate = random.randint(10, 15)
        img = dilation_img(img, k_size=k_dilate)
        

    # Augment and transparent image
    img = create_transparent_image(img, threshold=100)
    img = change_color_transparent(img, color=color)

    # fill raw signature to blur background
    shadow_trans.paste(img, (0,0), img)

    return shadow_trans



if __name__ == "__main__":

    img = Image.open("/media/geneous/e3359753-18af-45fb-b3d4-c5a0ced00e0c/Techainer/OCR/sig_ReID/data_collection/prepared_public_data/40_3_4_5_15_signature/Genuine/signature_chu_ky_thu_nhat_(specimen_i).jpg")
    img = Image.open("/media/geneous/e3359753-18af-45fb-b3d4-c5a0ced00e0c/Techainer/OCR/sig_ReID/data_collection/prepared_public_data/40_3_4_5_15_signature/Genuine/Same_signature_3_4_5_15_2.jpg")

    # temp = blur_signature_std(img, k_svd=20, color=[0, (0, 100), (130, 255)])
    # img = create_transparent_image(img, threshold=130)
    # img.show()
    # temp = svd_compress(img, k=30)

    # temp = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
    # cv2.imshow("",temp)
    # cv2.waitKey(0)
    # temp.show()
