###################################################
#
#   Script to pre-process the original imgs
#
##################################################


import numpy as np
from PIL import Image
import cv2

from help_functions import *
import configparser

config = configparser.RawConfigParser()
config.read('./configuration.txt', encoding='utf-8')
dataset = config.get('public', 'dataset')

dataset_mean = float(config.get(dataset, 'dataset_mean'))
dataset_std = float(config.get(dataset, 'dataset_std'))


def show_image(img):
    print(np.max(img), np.min(img))
    img = Image.fromarray(img)
    img.show()


# pre processing for grey_scale images(use for both training and testing!)
def my_PreProc(data):
    assert (len(data.shape) == 4)
    assert (data.shape[1] == 3)  # Use the original images
    # black-white conversion
    train_imgs = rgb2gray(data)
    # show_image(train_imgs[0][0])

    # my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    # show_image(train_imgs[0][0])
    train_imgs = clahe_equalized(train_imgs)
    # show_image(train_imgs[0][0])
    train_imgs = adjust_gamma(train_imgs, 1.2)
    # show_image(train_imgs[0][0])
    return train_imgs


# pre processing for color_scale images(use for both training and testing!)
def color_PreProc(data):
    assert (len(data.shape) == 4)
    assert (data.shape[1] == 3)  # Use the original images
    train_imgs = dataset_normalized(data)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    return train_imgs


# ============================================================
# ========= PRE PROCESSING FUNCTIONS ========================#
# ============================================================

# ==== histogram equalization
def histo_equalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = cv2.equalizeHist(np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# adaptive histogram equalization is used.
# In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV).
# Then each of these blocks are histogram equalized as usual.
# So in a small area, histogram would confine to a small region (unless there is noise).
# If noise is there, it will be amplified. To avoid this, contrast limiting is applied.
# If any histogram bin is above the specified contrast limit (by default 40 in OpenCV),
# those pixels are clipped and distributed uniformly to other bins before applying histogram equalization.
# After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    # create a CLAHE object (Arguments are optional).
    imgs_equalized = np.empty(imgs.shape)
    if imgs.shape[1] == 1:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for i in range(imgs.shape[0]):
            imgs_equalized[i, 0] = clahe.apply(np.array(imgs[i, 0], dtype=np.uint8))
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for channel in range(imgs.shape[1]):
            img = imgs[:, channel:(channel + 1):, :, ]
            img_equalized = np.empty(img.shape)
            for i in range(imgs.shape[0]):
                img_equalized[i, 0] = clahe.apply(np.array(img[i, 0], dtype=np.uint8))
            imgs_equalized[:, channel:(channel + 1):, :, ] = img_equalized

    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    if imgs.shape[1] == 1:
        # imgs_normalized = np.empty(imgs.shape)
        # imgs_std = np.std(imgs)
        # imgs_mean = np.mean(imgs)
        imgs_std = dataset_std
        imgs_mean = dataset_mean
        imgs_normalized = (imgs - imgs_mean) / imgs_std
        for i in range(imgs.shape[0]):
            imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
                np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))) * 255
        return imgs_normalized
    else:
        imgs_normalized = np.empty(imgs.shape)
        for channel in range(imgs.shape[1]):
            img = imgs[:, channel:(channel + 1), :, :]
            # img_normalized = np.empty(img.shape)
            img_std = np.std(img)
            img_mean = np.mean(img)
            img_normalized = (img - img_mean) / img_std
            for i in range(imgs.shape[0]):
                img_normalized[i] = ((img_normalized[i] - np.min(img_normalized[i])) / (
                    np.max(img_normalized[i]) - np.min(img_normalized[i]))) * 255
            imgs_normalized[:, channel: (channel + 1), :, :] = img_normalized

        return imgs_normalized


# adjust gamma
def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape) == 4)  # 4D arrays
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    if imgs.shape[1] == 1:
        for i in range(imgs.shape[0]):
            new_imgs[i, 0] = cv2.LUT(np.array(imgs[i, 0], dtype=np.uint8), table)
    else:
        for channel in range(imgs.shape[1]):
            img = imgs[:, channel:(channel + 1), :, :]
            new_img = np.empty(img.shape)
            for i in range(imgs.shape[0]):
                new_img[i, 0] = cv2.LUT(np.array(img[i, 0], dtype=np.uint8), table)
            new_imgs[:, channel:(channel + 1), :, :] = img
    return new_imgs
