import h5py
import numpy as np


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:
        return f['image'][()]


#
# def load_hdf5(infile):
#     with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
#         return f["image"][()]

def cal_mean_var(dataset):
    url = "./temp2/DRIVE_datasets_training_testing/DRIVE_dataset_imgs_train.hdf5".replace('DRIVE', dataset)
    image = load_hdf5(url)
    print(image.shape)

    ave_r = np.average(image[:, 0:1, :, :])
    ave_g = np.average(image[:, 1:2, :, :])
    ave_b = np.average(image[:, 2:3, :, :])
    ave = np.average(image)

    std_r = np.std(image[:, 0:1, :, :])
    std_g = np.std(image[:, 1:2, :, :])
    std_b = np.std(image[:, 2:3, :, :])
    std = np.std(image)

    print("average is: ", ave_r, ave_g, ave_b, ave)
    print("standard deviation is: ", std_r, std_g, std_b, std)


if __name__ == '__main__':
    cal_mean_var('CHASEDB1')
