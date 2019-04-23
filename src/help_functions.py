import h5py
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import pickle
import os
from pre_processing import my_PreProc

plt.switch_backend('agg')


# load images from hdf5 files
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
            # file = files[i]
            file = "shenqi_Prediction_"+str(i)+".png"
            img = Image.open(imgs_dir + file)
            img = np.array(img)
            print(imgs_dir + file, img.shape, img.max(), img.min())

            if np.shape(img.shape)[0] == 2:
                img = img.reshape((img.shape[0], img.shape[1], 1))
            imgs[i] = img

        imgs = np.transpose(imgs, (0, 3, 1, 2))
    return imgs


def prepare_data(urls_original, urls_save):
    for i in range(urls_original.__len__()):
        img = get_datasets(urls_original[i])
        if np.max(img) <= 1:
            img = img * 255
        print("the out is：", np.shape(img), np.max(img), np.min(img))

        write_hdf5(img, urls_save[i])


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
        x_center = np.random.randint(0 + int(patch_w / 2), (train_mask.shape[3] - int(patch_w / 2)) / mul,
                                     patch_per_img)
        y_center = np.random.randint(0 + int(patch_h / 2), int((train_mask.shape[2] - int(patch_h / 2)) / mul),
                                     patch_per_img)

        print(x_center)

        tu = sorted([(np.sum(x_center == i), i) for i in set(x_center.flat)])
        print('个数最多元素为 {1} 有 {0} 个'.format(*tu[-1]))

        x_center = x_center * mul
        y_center = y_center * mul
        print(x_center.shape, y_center.shape)

        for i in range(patch_per_img):
            temp = [x_center[i], y_center[i]]
            list_.append(temp)
        list1.append(list_)
    print(zero, one)
    print(np.shape(list1))
    return list1


def get_training_nodes(DRIVE_train_mask, patch_height, patch_width, N_subimgs, save_url):
    train_mask = load_hdf5(DRIVE_train_mask)
    train_mask = train_mask / 255
    print(np.min(train_mask), np.max(train_mask))
    assert (np.min(train_mask) == 0 and np.max(train_mask) == 1)
    patch = extract_random_in(train_mask, patch_height, patch_width, N_subimgs)
    save_node(patch, save_url)


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
        self.imgs = my_PreProc(self.imgs)
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
            for j in range(int(training_shape[1] / 10)):
                urls = self.get_url(int(j), length)
                # print(int(j),urls,length)
                # stop =input()

                for url in urls:
                    if not os.path.exists(img_temp + "/" + url):
                        os.mkdir(img_temp + "/" + url)
                    if not os.path.exists(label_temp + "/" + url):
                        os.mkdir(label_temp + "/" + url)

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
                    paths = self.get_url(int(iter_tot / 10), length - 1)
                    # print(paths,length,int(iter_tot / 10))
                    # stop =input()
                    save_img_url = paths[-1] + str(iter_tot % 10) + ".hdf5"
                    save_label_url = paths[-1] + str(iter_tot % 10) + ".hdf5"
                write_hdf5(patch, save_imgs_url + save_img_url)
                write_hdf5(patch_gtruth, save_labels_url + save_label_url)

                iter_tot += 1  # total
                self.names.append(["/train_patches/" + num_of_img + "/" + save_img_url,
                                   "/train_labels/" + num_of_img + "/" + save_img_url])


def get_model(shape_input, config, save_url):
    n_ch = shape_input[1]
    patch_height = shape_input[2]
    patch_width = shape_input[3]

    network = __import__(config.network_file)
    get_net = getattr(network, config.get_net)

    model = get_net(n_ch, patch_height, patch_width)
    print("Check: final output of the network:", model.output_shape)
    json_string = model.to_json()
    open(save_url, 'w').write(json_string)

    return model


# visualize image (as PIL image, NOT as matplotlib!)
def visualize(data, filename):
    assert (len(data.shape) == 3)  # height*width*channels
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))  # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    return img


# group a set of images row per columns
def group_images(data, per_row):
    assert data.shape[0] % per_row == 0
    assert (data.shape[1] == 1 or data.shape[1] == 3)
    data = np.transpose(data, (0, 2, 3, 1))  # corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0] / per_row)):
        stripe = data[i * per_row]
        for k in range(i * per_row + 1, i * per_row + per_row):
            stripe = np.concatenate((stripe, data[k]), axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1, len(all_stripe)):
        totimg = np.concatenate((totimg, all_stripe[i]), axis=0)
    return totimg


def save_sample(patches_imgs_train, save_url):
    # ========= Save a sample of what you're feeding to the neural network ==========
    N_sample = min(patches_imgs_train.shape[0], 40)
    visualize(group_images(patches_imgs_train[0:N_sample, :, :, :], 5), save_url)


def show_img(img):
    imgs = []
    for i in range(img.shape[0]):
        temp = img[i][0]
        imgs.extend(temp)
    print(np.shape(imgs))
    Image.fromarray(np.array(imgs)).show()
