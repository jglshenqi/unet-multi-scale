# ==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
# ============================================================

import os
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


# ------------Path of the images --------------------------------------------------------------
dataset = "CHASEDB1"
# train
original_imgs_train = "./" + dataset + "/training/images/"
groundTruth_imgs_train = "./" + dataset + "/training/manual/"
borderMasks_imgs_train = "./" + dataset + "/training/mask/"
# test
original_imgs_test = "./" + dataset + "/test/images/"
groundTruth_imgs_test = "./" + dataset + "/test/manual/"
borderMasks_imgs_test = "./" + dataset + "/test/mask/"
# ---------------------------------------------------------------------------------------------

channels = 3
train_img = 14
test_img = 14
height = 960
width = 999
piece = [2, 2]
dataset_path = "./" + dataset + "_datasets_training_testing/"


def get_datasets(imgs_dir, groundTruth_dir, borderMasks_dir, train_test="null"):
    if train_test == "train":
        Nimgs = train_img
    elif train_test == "test":
        Nimgs = test_img

    piece_w = int(piece[0])
    piece_h = int(piece[1])
    new_w = int(width / piece[0])
    new_h = int(height / piece[0])
    Nimgs = Nimgs * piece_w * piece_h
    imgs = np.empty((Nimgs, new_h, new_w, channels))
    groundTruth = np.empty((Nimgs, new_h, new_w))
    border_masks = np.empty((Nimgs, new_h, new_w))
    for path, subdirs, files in os.walk(imgs_dir):  # list all files, directories in the path
        for i in range(len(files)):
            # print(path,subdirs,files,i)
            # original
            leng = files[i].__len__()
            print("original image: " + files[i])
            img = Image.open(imgs_dir + files[i])
            groundTruth_name = files[i][0:(leng - 4)] + "_1stHO.png"
            print("ground truth name: " + groundTruth_name)
            border_masks_name = files[i][0:leng - 4] + ".jpg"
            print("border masks name: " + border_masks_name)

            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            b_mask = b_mask.convert('L')
            print(np.asarray(img).shape, np.asarray(g_truth).shape)
            for h in range(piece[1]):
                for w in range(piece[0]):
                    number = i * piece_h * piece_w + h * piece_w + w
                    imgs[number] = np.asarray(img)[h * new_h:(h + 1) * new_h, w * new_w:(w + 1) * new_w, :]
                    groundTruth[number] = np.asarray(g_truth)[h * new_h:(h + 1) * new_h, w * new_w:(w + 1) * new_w]
                    border_masks[number] = np.asarray(b_mask)[h * new_h:(h + 1) * new_h, w * new_w:(w + 1) * new_w]

    print("imgs max: " + str(np.max(imgs)))
    print("imgs min: " + str(np.min(imgs)))
    # assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    # assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    # reshaping for my standard tensors
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    print(imgs.shape, Nimgs, channels, new_h, new_w)
    assert (imgs.shape == (Nimgs, channels, new_h, new_w))
    groundTruth = np.reshape(groundTruth, (Nimgs, 1, new_h, new_w))
    border_masks = np.reshape(border_masks, (Nimgs, 1, new_h, new_w))
    assert (groundTruth.shape == (Nimgs, 1, new_h, new_w))
    assert (border_masks.shape == (Nimgs, 1, new_h, new_w))
    return imgs, groundTruth, border_masks


if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
# getting the training datasets
imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train, groundTruth_imgs_train,
                                                                 borderMasks_imgs_train, "train")
print("saving train datasets")
write_hdf5(imgs_train, dataset_path + dataset + "_dataset_imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + dataset + "_dataset_groundTruth_train.hdf5")
write_hdf5(border_masks_train, dataset_path + dataset + "_dataset_borderMasks_train.hdf5")

# getting the testing datasets
imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test, groundTruth_imgs_test,
                                                              borderMasks_imgs_test, "test")
print(np.shape(imgs_test))
print("saving test datasets")
write_hdf5(imgs_test, dataset_path + dataset + "_dataset_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + dataset + "_dataset_groundTruth_test.hdf5")
write_hdf5(border_masks_test, dataset_path + dataset + "_dataset_borderMasks_test.hdf5")
