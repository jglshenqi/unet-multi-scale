import configparser
from help_functions import *
from extract_patches import *

config = configparser.RawConfigParser()
config.read('../configuration.txt')
dataset = "STARE"
# patch to the datasets
path_data = "." + config.get('data paths', 'path_local').replace("DRIVE", dataset)
n_subings = 180000

save_url = "../dataset/DRIVE_datasets_training_testing/train_patch.pickle".replace("DRIVE", dataset)


def extract_random_in(train_mask, patch_h, patch_w, N_patches):
    print("mask shape is:", train_mask.shape)

    patch_per_img = int(N_patches / train_mask.shape[0])  # N_patches equally divided in the full images
    iter_tot = 0  # iter over the total numbe rof patches (N_patches)
    list = []
    one = 0
    zero = 0
    for i in range(train_mask.shape[0]):  # loop over the full images
        k = 0
        list_ = []
        while k < patch_per_img:
            x_center = random.randint(0 + int(patch_w / 2), train_mask.shape[3] - int(patch_w / 2))
            y_center = random.randint(0 + int(patch_h / 2), train_mask.shape[2] - int(patch_h / 2))

            temp = [x_center, y_center]

            patch = train_mask[i, :, y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                    x_center - int(patch_w / 2):x_center + int(patch_w / 2)]

            if np.min(patch) < 0:
                zero = zero + 1
            else:
                one = one + 1
                list_.append(temp)

                iter_tot += 1  # total
                k += 1  # per full_img

        list.append(list_)
    print(zero, one)
    save_node(list, save_url)
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
