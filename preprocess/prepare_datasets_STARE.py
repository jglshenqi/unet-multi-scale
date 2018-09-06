import os
import h5py
import numpy as np
from PIL import Image
from public import *
import time


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def get_datasets(imgs_dir):
    print(imgs_dir)
    imgs = None
    for path, subdirs, files in os.walk(imgs_dir):

        print(path, subdirs, files)
        img = Image.open(path + files[0])
        size = np.shape(img)
        print(np.shape(img))

        if np.shape(size)[0] == 2:
            imgs = np.zeros((len(files), size[0], size[1], 1))
        else:
            imgs = np.zeros((len(files), size[0], size[1], size[2]))

        for i in range(len(files)):
            file = files[i]

            img = Image.open(imgs_dir + file)
            img = np.array(img)
            print(imgs_dir + file, img.shape, img.max(), img.min())

            if np.shape(img.shape)[0] == 2:
                img = img.reshape((img.shape[0], img.shape[1], 1))
            imgs[i] = img

        imgs = np.transpose(imgs, (0, 3, 1, 2))
        # show_img(imgs)

    return imgs


def prepare_data(urls_original, urls_save):
    for i in range(urls_original.__len__()):
        start_time = time.time()
        img = get_datasets(urls_original[i])
        if np.max(img)==1:
            img = img*255
        print("the out is：", np.shape(img), np.max(img), np.min(img))
        print(time.time() - start_time)

        write_hdf5(img, urls_save[i])
        print(time.time() - start_time)


def example(dataset):
    root = './temp/' + dataset
    dataset_path = root + "_datasets_training_testing/"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    # train
    original_imgs_train = root + "/training/images/"
    groundTruth_imgs_train = root + "/training/manual/"
    borderMasks_imgs_train = root + "/training/mask/"

    imgs_train_save = dataset_path + dataset + "_dataset_imgs_train.hdf5"
    gtruth__train_save = dataset_path + dataset + "_dataset_groundTruth_train.hdf5"
    mask_train_save = dataset_path + dataset + "_dataset_borderMasks_train.hdf5"
    url_original_train = [original_imgs_train, groundTruth_imgs_train, borderMasks_imgs_train]
    url_save_train = [imgs_train_save, gtruth__train_save, mask_train_save]

    # test
    original_imgs_test = root + "/test/images/"
    groundTruth_imgs_test = root + "/test/manual/"
    borderMasks_imgs_test = root + "/test/mask/"

    imgs_test_save = dataset_path + dataset + "_dataset_imgs_test.hdf5"
    gtruth_test_save = dataset_path + dataset + "_dataset_groundTruth_test.hdf5"
    mask_test_save = dataset_path + dataset + "_dataset_borderMasks_test.hdf5"

    url_original_test = [original_imgs_test, groundTruth_imgs_test, borderMasks_imgs_test]
    url_save_test = [imgs_test_save, gtruth_test_save, mask_test_save]

    prepare_data(url_original_train, url_save_train)
    prepare_data(url_original_test, url_save_test)

    # prepare_data(url_original_train[2], url_save_train[2])
    # prepare_data(url_original_test[2], url_save_test[2])

    # url_original = ["./shenqi/0/"]
    # url_save = ["./shenqi/DRIVE_unet.hdf5"]
    #
    # prepare_data(url_original, url_save)


# # prepare_data(["./shenqi/"], ["./HRF/1.hdf5"])
dataset = 'CHASEDB1'
example(dataset)

# prepare_data(["./shenqi/"], ["./shenqi/1.hdf5"])
