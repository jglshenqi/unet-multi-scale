import get_config
import time
import os
import numpy as np
from help_functions import write_hdf5, load_hdf5, load_node, get_training_nodes, get_datasets, draw_patch
from extract_patches import get_data_training

config = get_config.get_config()


def prepare_dataset():
    dataset = "DRIVE"
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


def get_patch():
    dataset = "DRIVE"
    path_data = './temp/DRIVE_datasets_training_testing/'.replace("DRIVE", dataset)
    n_subings = 200
    patch_size = [48, 48]
    save_url = path_data + "train_patch" + str(patch_size[0]) + "_" + str(n_subings) + ".pickle"

    get_training_nodes(
        DRIVE_train_mask=path_data + 'DRIVE_dataset_borderMasks_train.hdf5'.replace("DRIVE", dataset),
        patch_height=patch_size[0],
        patch_width=patch_size[1],
        N_subimgs=n_subings,
        save_url=save_url)

    a = load_node(save_url)
    print(np.shape(a))


def draw_patches():
    dataset = "DRIVE"
    root = "./temp/" + dataset + "_datasets_training_testing" + "/"

    train_imgs_original = root + dataset + "_dataset_imgs_train.hdf5"
    train_groudTruth = root + dataset + "_dataset_groundTruth_train.hdf5"
    patch_size = [48, 48]
    train_coordinate = root + "train_patch48_200.pickle"
    training_format = 1
    train_patches = root + "train_patch48_200.hdf5"
    groundtruth_patches = root + "groundtruth_patch48_200.hdf5"

    patches_imgs_train, patches_masks_train = get_data_training(
        DRIVE_train_imgs_original=train_imgs_original,
        DRIVE_train_groudTruth=train_groudTruth,
        patch_height=patch_size[0],
        patch_width=patch_size[1],
        train_coordinate=train_coordinate,
        training_format=training_format)

    patches_masks_train = np.array(patches_masks_train)
    patches_imgs_train = np.array(patches_imgs_train)

    write_hdf5(patches_imgs_train, train_patches)
    write_hdf5(patches_masks_train, groundtruth_patches)

    print(np.shape(load_hdf5(train_patches)))
    print(np.shape(load_hdf5(groundtruth_patches)))
    print("write over")


def draw_single_patch():
    dataset = "DRIVE"
    root = "./temp/" + dataset + "_datasets_training_testing"
    patch_h = 48
    patch_num = 200
    temp = draw_patch(root, dataset, patch_h, patch_num)
    temp.save_patch()

    # temp.change(patch_h=96, patch_num=150000)
    # temp.save_patch()


if __name__ == '__main__':
    print("here are preprocessing files")

    # transform dataset from images to hdf5 files
    # prepare_dataset()

    # get the training patches
    # get_patch()

    # draw the training patches to hdf5
    # draw_patches()

    # draw a sigle training patch to hdf5
    # draw_single_patch()
