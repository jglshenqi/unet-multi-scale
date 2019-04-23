import numpy as np
import cv2
import h5py
import pickle
import os
import random


def test_patch():
    # dataset = "HRF"
    # ori_path = "../dataset/DRIVE/DRIVE_datasets_training_testing".replace("DRIVE", dataset)
    # patch_h = 48
    # patch_num = 200
    # a = load_hdf5(ori_path+"/train_patch_"+str(patch_h)+"_"+str(patch_num)+"/train_patch_0.hdf5")
    # c = load_hdf5(
    #     "./dataset/DRIVE/DRIVE_datasets_training_testing/train_patch_48_200/train_patch_0.hdf5")
    # a = load_node("./temp/train_patch_test")
    # b = load_node("./temp/train_patch_w")
    # print(np.shape(b), np.shape(a))

    # b = b[0][0]
    # print(a[0][0])
    # a = load_node("./temp/1.pickle")
    # b = load_node("./temp/2.pickle")
    # print(a,b)
    # print(c[0][0])
    dataset = 'DRIVE'
    path = "../dataset/HRF/HRF_datasets_training_testing".replace("HRF", dataset)

    generate_train_arrays_from_file3(path + "/train_patch_48_200", 0.9)


def test():
    dataset = "CHASEDB1"
    ori_path = "../dataset/DRIVE/DRIVE_datasets_training_testing".replace("DRIVE", dataset)
    patch_h = 64
    patch_num = 200000
    temp = draw_patch(ori_path, dataset, patch_h, patch_num)
    print(ori_path)
    temp.save_patch()

    # temp.change(patch_h=96, patch_num=150000)
    # temp.save_patch()
    # temp.change(patch_h=128, patch_num=150000)
    # temp.save_patch()
    # temp.change(patch_h=128, patch_num=300000)
    # temp.save_patch()
    # temp.change(patch_h=96, patch_num=300000)
    # temp.save_patch()
    # temp.change(patch_h=48, patch_num=300000)
    # temp.save_patch()
    # temp.change(patch_h=48, patch_num=450000)
    # temp.save_patch()
    # temp.change(patch_h=96, patch_num=450000)
    # temp.save_patch()
    # temp.change(patch_h=128, patch_num=450000)
    # temp.save_patch()

    # temp = load_node(ori_path + "/" + "train_patch_" + str(patch_h) + "_" + str(patch_num) + "/patch_names.pickle")
    # print(np.shape(temp))


def read_files(path, patch_name):
    x = []
    y = []
    for line in patch_name:
        print(line[0])
        x_url = path + "/train_patches/" + line[0] + ".hdf5"
        y_url = path + "/train_labels/" + line[1] + ".hdf5"
        if x == []:
            x = np.array(load_hdf5(x_url))
            y = np.array(load_hdf5(y_url))
        else:
            x = np.concatenate((x, np.array(load_hdf5(x_url))))
            y = np.concatenate((y, np.array(load_hdf5(y_url))))
    x = np.concatenate((x, y), axis=1)
    return x


def generate_train_arrays_from_file3(path, validation_split):
    patch_names = load_node(path + "/patch_names.pickle")
    # print(np.shape(patch_names))

    patch_names = patch_names[:int(len(patch_names) * validation_split)]
    # print(np.shape(patch_names))
    temp = read_files(path, patch_names)
    print("load finish")

    random.shuffle(temp)
    print("random finish")

    save_patches_url = path + "/train_patches_random"
    save_labels_url = path + "/train_labels_random"

    if not os.path.exists(save_patches_url):
        os.mkdir(save_patches_url)
    if not os.path.exists(save_labels_url):
        os.mkdir(save_labels_url)

    num_per_img = len(temp) / len(patch_names)
    print(temp.shape, num_per_img)
    num = 0

    for line in patch_names:
        up = int((num + 1) * num_per_img)
        down = int(num * num_per_img)
        # a = temp[down:up, 0:1, :, :]
        # print(np.shape(a), up, down)
        write_hdf5(temp[down:up, 0:1, :, :], save_patches_url + "/" + line[0] + ".hdf5")
        write_hdf5(temp[down:up, 1:2, :, :], save_labels_url + "/" + line[1] + ".hdf5")
        print(line[0])
        num = num + 1


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["image"][()]


# write images into hdf5 files
def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def save_node(list_, url):
    with open(url, 'wb') as f:
        pickle.dump(list_, f)


def load_node(url):
    with open(url, 'rb') as f:
        list_ = pickle.load(f)
    return list_


class draw_patch():
    def __init__(self, ori_url, dataset, patch_h, patch_num):
        self.ori_url = ori_url
        self.dataset = dataset
        self.patch_h = patch_h
        self.patch_num = patch_num
        self.imgs_url = self.ori_url + "/" + self.dataset + "_dataset_imgs_train.hdf5"
        self.labels_url = self.ori_url + "/" + self.dataset + "_dataset_groundTruth_train.hdf5"
        self.coordinate = self.ori_url + "/train_patch" + str(self.patch_h) + "_" + str(self.patch_num) + ".pickle"
        self.save_url = self.ori_url + "/train_patch_" + str(self.patch_h) + "_" + str(self.patch_num)
        self.names = []
        self.imgs = load_hdf5(self.imgs_url)
        self.labels = load_hdf5(self.labels_url)

    def change(self, patch_h, patch_num):
        self.patch_h = patch_h
        self.patch_num = patch_num
        self.imgs_url = self.ori_url + "/" + self.dataset + "_dataset_imgs_train.hdf5"
        self.labels_url = self.ori_url + "/" + self.dataset + "_dataset_groundTruth_train.hdf5"
        self.coordinate = self.ori_url + "/train_patch" + str(self.patch_h) + "_" + str(self.patch_num) + ".pickle"
        self.save_url = self.ori_url + "/train_patch_" + str(self.patch_h) + "_" + str(self.patch_num)
        self.names = []
        self.imgs = load_hdf5(self.imgs_url)
        self.labels = load_hdf5(self.labels_url)

    def save_patch(self):
        print("========Now dealing with ", str(self.patch_h) + "_" + str(self.patch_num) + "========")
        self.my_PreProc()
        print("Preprocessing ready")

        self.imgs /= 255
        self.labels /= 255

        # print(self.imgs.shape,self.imgs[1][0][22])

        print("train images shape:", self.imgs.shape,
              "train images range (min-max): " + str(np.min(self.imgs)) + ' - ' + str(np.max(self.imgs)))
        print("train gtruth shape:", self.labels.shape,
              "train gtruth range (min-max): " + str(np.min(self.labels)) + ' - ' + str(np.max(self.labels)))

        self.mkpath()

        print("Make path ready")

        self.extract_coordinate()

        print("Exact ready")

        self.save_names()

        # print(self.patches_imgs.shape, self.patches_labels.shape)

    def save_names(self):
        save_node(self.names, self.save_url + "/patch_names.pickle")

    def get_url(self, num, length):
        num = str(num)
        url = ""
        urls = []
        while len(num) < length:
            num = "0" + num
        for temp in num:
            url = url + temp + "/"
            urls.append(url)
        return urls

    def mkpath(self):
        img_url = self.save_url + "/train_patches/"
        label_url = self.save_url + "/train_labels/"
        if not os.path.exists(self.save_url):
            os.mkdir(self.save_url)
        if not os.path.exists(label_url):
            os.mkdir(label_url)
        if not os.path.exists(img_url):
            os.mkdir(img_url)
        training_node = load_node(self.coordinate)
        training_shape = np.shape(training_node)
        # training_shape = (20, 8000, 2)
        print(self.patch_num, self.names, training_shape)

        for i in range(training_shape[0]):
            num = training_shape[1]
            if i < 10:
                temp = "0" + str(i)
            else:
                temp = str(i)
            img_temp = img_url + temp
            label_temp = label_url + temp
            if not os.path.exists(img_temp):
                os.mkdir(img_temp)
            if not os.path.exists(label_temp):
                os.mkdir(label_temp)

            paths = np.arange(0, int(num / 10)).tolist()
            if paths == [0]:
                continue
            length = len(str(np.max(paths)))
            for j in range(int(training_shape[1]/10)):
                urls = self.get_url(int(j), length)
                # print(int(j),urls,length)
                # stop =input()

                for url in urls:
                    if not os.path.exists(img_temp + "/" + url):
                        os.mkdir(img_temp + "/" + url)
                    if not os.path.exists(label_temp + "/" + url):
                        os.mkdir(label_temp + "/" + url)

    def my_PreProc(self):
        assert (len(self.imgs.shape) == 4)
        assert (self.imgs.shape[1] == 3)  # Use the original images
        # black-white conversion
        self.rgb2gray()

        self.dataset_normalized()
        # show_image(train_imgs[0][0])
        self.clahe_equalized()
        # show_image(train_imgs[0][0])

        self.adjust_gamma(1.2)
        # show_image(train_imgs[0][0])

    def rgb2gray(self):
        assert (len(self.imgs.shape) == 4)  # 4D arrays
        assert (self.imgs.shape[1] == 3)
        imgs_grey = self.imgs[:, 0, :, :] * 0.299 + self.imgs[:, 1, :, :] * 0.587 + self.imgs[:, 2, :, :] * 0.114
        imgs_grey = np.reshape(imgs_grey, (self.imgs.shape[0], 1, self.imgs.shape[2], self.imgs.shape[3]))
        self.imgs = imgs_grey

    def dataset_normalized(self):
        assert (len(self.imgs.shape) == 4)  # 4D arrays
        # imgs_normalized = np.empty(imgs.shape)
        # imgs_std = np.std(self.imgs)
        # imgs_mean = np.mean(self.imgs)

        imgs_mean = 80.83841938578134
        imgs_std = 76.43214153089184
        # imgs_std = dataset_std
        # imgs_mean = dataset_mean
        imgs_normalized = (self.imgs - imgs_mean) / imgs_std
        # print(np.std(imgs_normalized), np.mean(imgs_normalized))
        for i in range(self.imgs.shape[0]):
            imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
                np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))) * 255
        self.imgs = imgs_normalized

    def adjust_gamma(self, gamma=1.0):
        assert (len(self.imgs.shape) == 4)  # 4D arrays
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        new_imgs = np.empty(self.imgs.shape)
        for i in range(self.imgs.shape[0]):
            new_imgs[i, 0] = cv2.LUT(np.array(self.imgs[i, 0], dtype=np.uint8), table)
        self.imgs = new_imgs

    def clahe_equalized(self):
        assert (len(self.imgs.shape) == 4)  # 4D arrays
        # create a CLAHE object (Arguments are optional).
        imgs_equalized = np.empty(self.imgs.shape)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for i in range(self.imgs.shape[0]):
            imgs_equalized[i, 0] = clahe.apply(np.array(self.imgs[i, 0], dtype=np.uint8))
        self.imgs = imgs_equalized

    def extract_coordinate(self):
        training_node = load_node(self.coordinate)
        training_shape = np.shape(training_node)
        assert (len(self.imgs.shape) == 4 and len(self.labels.shape) == 4)  # 4D arrays
        assert (self.imgs.shape[1] == 1 or self.imgs.shape[1] == 3)  # check the channel is 1 or 3
        assert (self.labels.shape[1] == 1)  # masks only black and white
        assert (self.imgs.shape[2] == self.labels.shape[2] and self.imgs.shape[3] == self.labels.shape[3])
        patch_h = self.patch_h
        patch_w = self.patch_h
        # N_patches equally divided in the full images
        patch_per_img = int(training_shape[0] * training_shape[1] / self.imgs.shape[0])
        print("patches per full image: " + str(patch_per_img))

        for i in range(training_shape[0]):  # loop over the full images
            if i < 10:
                num_of_img = "0" + str(i)
            else:
                num_of_img = str(i)
            print("deal with the image:", num_of_img)
            iter_tot = 0
            save_imgs_url = self.save_url + "/train_patches/" + num_of_img + "/"
            save_labels_url = self.save_url + "/train_labels/" + num_of_img + "/"
            length = len(str(training_shape[1] - 1))
            while iter_tot < training_shape[1]:
                x_center = training_node[i][iter_tot][0]
                y_center = training_node[i][iter_tot][1]

                patch = self.imgs[i, :, y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                        x_center - int(patch_w / 2):x_center + int(patch_w / 2)]
                patch_gtruth = self.labels[i, :, y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                               x_center - int(patch_w / 2):x_center + int(patch_w / 2)]
                if training_shape[1] <= 10:
                    save_img_url = str(iter_tot) + ".hdf5"
                    save_label_url = str(iter_tot) + ".hdf5"
                else:
                    paths = self.get_url(int(iter_tot / 10), length-1)
                    # print(paths,length,int(iter_tot / 10))
                    # stop =input()
                    save_img_url = paths[-1] + str(iter_tot % 10) + ".hdf5"
                    save_label_url = paths[-1] + str(iter_tot % 10) + ".hdf5"
                write_hdf5(patch, save_imgs_url + save_img_url)
                write_hdf5(patch_gtruth, save_labels_url + save_label_url)

                iter_tot += 1  # total
                self.names.append(["/train_patches/" + num_of_img + "/" + save_img_url,
                                   "/train_labels/" + num_of_img + "/" + save_img_url])
        # print(self.names)
        # save_node(self.names, self.save_url + "patch_names.pickle")


if __name__ == '__main__':
    test()

    # test_patch()
