import configparser
# from help_functions import *
# from extract_patches import *
import random
import numpy as np
import h5py
import pickle
from PIL import Image

def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["image"][()]

def load_node(url):
    with open(url, 'rb') as f:
        list_ = pickle.load(f)
    return list_

def save_node(list_, url):
    with open(url, 'wb') as f:
        pickle.dump(list_, f)

config = configparser.RawConfigParser()
config.read('../configuration.txt')
dataset = "HRF"
# patch to the datasets
path_data = "." + config.get('data paths', 'path_local').replace("DRIVE", dataset)
n_subings = 180000

save_url = "../DRIVE_datasets_training_testing/train_patch.pickle".replace("DRIVE", dataset)


def extract_random_in(train_mask, patch_h, patch_w, N_patches):
    print("mask shape is:", train_mask.shape)

    patch_per_img = int(N_patches / train_mask.shape[0])  # N_patches equally divided in the full images
    iter_tot = 0  # iter over the total numbe rof patches (N_patches)
    list1 = []
    list_2 = []
    one = 0
    zero = 0
    for i in range(train_mask.shape[0]):  # loop over the full images
        k = 0
        list_ = []

        mul = 1
        x_center = np.random.randint(0 + int(patch_w / 2), (train_mask.shape[3] - int(patch_w / 2)) / mul,patch_per_img)
        y_center = np.random.randint(0 + int(patch_h / 2), int((train_mask.shape[2] - int(patch_h / 2)) / mul),patch_per_img)

        print(x_center)

        tu = sorted([(np.sum(x_center == i), i) for i in set(x_center.flat)])
        print('个数最多元素为 {1} 有 {0} 个'.format(*tu[-1]))

        x_center = x_center*mul
        y_center = y_center*mul
        print(x_center.shape,y_center.shape)

        for i in range(patch_per_img):
            temp = [x_center[i],y_center[i]]
            list_.append(temp)

        # list_ = np.vstack((x_center,y_center))
        # list_ = list_.reshape((patch_per_img,2))
        # print(list_.shape)
        # list_ = list(list_)
        # while k < patch_per_img:
        #     mul = 3
        #     x_center = random.randint(0 + int(patch_w / 2), (train_mask.shape[3] - int(patch_w / 2)) / mul)
        #     y_center = random.randint(0 + int(patch_h / 2), int((train_mask.shape[2] - int(patch_h / 2)) / mul))
        #
        #     temp = [mul * x_center, mul * y_center]
        #
        #     patch = train_mask[i, :, y_center - int(patch_h / 2):y_center + int(patch_h / 2),
        #             x_center - int(patch_w / 2):x_center + int(patch_w / 2)]
        #
        #     # if np.min(patch) < 0:
        #     #     zero = zero + 1
        #     # else:
        #     #     one = one + 1
        #     #     list_.append(temp)
        #     #
        #     #     iter_tot += 1  # total
        #     #     k += 1  # per full_img
        #
        #
        #     if temp in list_2:
        #         zero = zero + 1
        #
        #     else:
        #         one = one + 1
        #         list_.append(temp)
        #
        #         list_2.append(temp)
        #         # list_2.append([temp[0] + 1, temp[1] + 1])
        #         # list_2.append([temp[0] + 1, temp[1] - 1])
        #         # list_2.append([temp[0] - 1, temp[1] - 1])
        #         # list_2.append([temp[0] - 1, temp[1] + 1])
        #         # list_2.append([temp[0], temp[1] + 1])
        #         # list_2.append([temp[0], temp[1] - 1])
        #         # list_2.append([temp[0] + 1, temp[1]])
        #         # list_2.append([temp[0] - 1, temp[1]])
        #
        #         iter_tot += 1  # total
        #         k += 1  # per full_img

            # if one % 1000 == 0 and zero != 0:
            #     print(zero, one)
            #     zero = 0

        list1.append(list_)
    print(zero, one)
    print(np.shape(list1))
    save_node(list1, save_url)
    print("end")


def get_data_training(DRIVE_train_mask,
                      patch_height,
                      patch_width,
                      N_subimgs):
    train_mask = load_hdf5(DRIVE_train_mask)
    train_mask = train_mask / 255
    print(np.min(train_mask), np.max(train_mask))
    assert (np.min(train_mask) == 0 and np.max(train_mask) == 1)
    extract_random_in(train_mask, patch_height, patch_width, N_subimgs)


def read_hdf5(url):
    img = load_hdf5(url)
    print(img.shape)
    for i in range(12, 16):
        Image.fromarray(img[i][0]).show()


if __name__ == '__main__':
    get_data_training(
        DRIVE_train_mask=path_data + config.get('data paths', 'train_border_masks').replace("DRIVE", dataset),
        patch_height=int(config.get('data attributes', 'patch_height')),
        patch_width=int(config.get('data attributes', 'patch_width')),
        N_subimgs=n_subings)

    # url = path_data + config.get('data paths', 'train_border_masks').replace("DRIVE", dataset)
    # read_hdf5(url)
    a = load_node(save_url)
    print(np.shape(a))
