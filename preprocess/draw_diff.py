from PIL import Image
import numpy as np
from skimage import io
import h5py
import os
from matplotlib import pyplot as plt
from skimage import filters
import cv2


def get_image(img, url_save):
    w = np.shape(img)[0]
    h = np.shape(img)[1]
    temp = np.zeros(shape=(w, h, 3)).tolist()
    for i in range(w):
        for j in range(h):
            if img[i][j] == 1:
                temp[i][j][0] = 255
            elif img[i][j] == 2:
                temp[i][j][1] = 255
            elif img[i][j] == 3:
                temp[i][j][0] = 255
                temp[i][j][1] = 255
                temp[i][j][2] = 255
    temp = np.array(temp) / 255
    io.imsave(url_save, temp)


def get_image2(img, url_save):
    w = np.shape(img)[0]
    h = np.shape(img)[1]
    temp = np.zeros(shape=(w, h, 3)).tolist()
    # img =np.reshape(img,newshape=(np.shape(img)[0]*np.shape(img)[1]))
    # draw_hist(img, 'distribution', 'x', 'y', 0, 8, 0.0, 30000)
    # stop =input()
    for i in range(w):
        for j in range(h):
            if img[i][j] == 1:
                temp[i][j][0] = 255
            elif img[i][j] == 2:
                temp[i][j][1] = 127
            elif img[i][j] == 4:
                temp[i][j][1] = 255
            elif img[i][j] == 7:
                temp[i][j][0] = 255
                temp[i][j][1] = 255
                temp[i][j][2] = 255
    temp = np.array(temp) / 255
    io.imsave(url_save, temp)


def draw_diff(url1, url2, url, threshold=0.5):
    img1 = load_hdf5(url1)
    img2 = load_hdf5(url2)

    img1 = np.array(img1)
    img2 = np.array(img2)

    img1 = img1 / 255 * 2
    img2 = img2 / 255

    img2[img2 > threshold] = 1
    img2[img2 <= threshold] = 0
    img = img1 + img2

    url = url + "_new_" + str(threshold)

    if not os.path.exists(url):
        os.mkdir(url)

    for i in range(img.shape[0]):
        print("preprocess image:" + str(i + 1))
        temp = img[i][0]
        get_image(temp, url + "/pred_" + str(i) + ".png")


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["image"][()]


def get_datasets(imgs_dir):
    print(imgs_dir)
    imgs = None
    for path, subdirs, files in os.walk(imgs_dir):

        img = Image.open(path + "/" + files[0])
        size = np.shape(img)
        print("the shape of images is:", np.shape(img))

        if np.array(size).shape[0] == 2:
            size = [size[0], size[1], 1]

        imgs = np.zeros((len(files), size[0], size[1], size[2]))

        print("")
        for i in range(len(files)):
            file = "shenqi" + "_Prediction_" + str(i) + ".png"

            print(imgs_dir + "/" + file, end="\r")
            img = Image.open(imgs_dir + "/" + file)
            img = np.array(img)

            if np.shape(img.shape)[0] == 2:
                img = img.reshape((img.shape[0], img.shape[1], 1))
            imgs[i] = img

        imgs = np.transpose(imgs, (0, 3, 1, 2))
        # show_img(imgs)
        print("")

    return imgs


def draw(dataset, net, threshold=0.5):
    print("draw pictures from dataset ", dataset, "and net ", net)
    url1 = "../dataset/DRIVE/DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_test.hdf5".replace("DRIVE",
                                                                                                          dataset)
    url2 = "./temp/DRIVE/unet3/temp.hdf5".replace("DRIVE", dataset).replace("unet3", net)
    url = "./temp/DRIVE/unet3".replace("DRIVE", dataset).replace("unet3", net)

    for path, subdirs, files in os.walk(url):
        if "temp.hdf5" not in files:
            temp = get_datasets(url)
            write_hdf5(temp, url + "/temp.hdf5")

    threshold = get_otsu_threshold(url2)
    draw_diff(url1, url2, url, threshold)


def draw_hist(myList, Title, Xlabel, Ylabel, Xmin, Xmax, Ymin, Ymax):
    plt.hist(myList, 100)
    plt.xlabel(Xlabel)
    plt.xlim(Xmin, Xmax)
    plt.ylabel(Ylabel)
    plt.ylim(Ymin, Ymax)
    plt.title(Title)
    plt.show()


def draw_distribution():
    dataset = "DRIVE"
    url_ori = "../dataset/DRIVE/DRIVE_datasets_training_testing".replace("DRIVE", dataset)

    train_imgs = load_hdf5(url_ori + "/" + "DRIVE_dataset_groundTruth_test.hdf5".replace("DRIVE", dataset)) / 255 * 2
    train_results = load_hdf5('./temp/DRIVE/br/temp.hdf5'.replace("DRIVE", dataset)) / 255

    train_threshold = np.array(train_results)

    train_threshold[train_threshold >= 0.5] = 1
    train_threshold[train_threshold < 0.5] = 0

    train_imgs = train_imgs + train_threshold

    for i in range(np.shape(train_imgs)[0]):

        train_img = train_imgs[i][0]
        train_result = train_results[i][0]
        distribution = []

        for i in range(584):
            for j in range(565):
                if train_img[i][j] == 2:
                    distribution.append(train_result[i][j])

        print(np.mean(distribution))

        draw_hist(distribution, 'distribution', 'x', 'y', 0, 0.5, 0.0, 100)  # 直方图展示
        stop = input()


def contrast(dataset, net1, net2, threshold=0.5):
    print("draw pictures from dataset ", dataset, "and contrast with ", net1, " and ", net2)
    url_gt = "../dataset/HRF/HRF_datasets_training_testing/HRF_dataset_groundTruth_test.hdf5".replace("HRF", dataset)
    url1 = "./temp/DRIVE/unet3/temp.hdf5".replace("DRIVE", dataset).replace("unet3", net1)
    url2 = "./temp/DRIVE/unet3/temp.hdf5".replace("DRIVE", dataset).replace("unet3", net2)

    url_save = "./temp/DRIVE/".replace('DRIVE', dataset) + net1 + " contrast " + net2
    if not os.path.exists(url_save):
        os.mkdir(url_save)

    for net in [net1, net2]:
        url_temp = "./temp/DRIVE/unet3".replace("DRIVE", dataset).replace("unet3", net)
        for path, subdirs, files in os.walk(url_temp):
            if "temp.hdf5" not in files:
                temp = get_datasets(url_temp)
                write_hdf5(temp, url_temp + "/temp.hdf5")

    img_gt = np.array(load_hdf5(url_gt)) / 255 * 4
    img1 = np.array(load_hdf5(url1)) / 255
    img2 = np.array(load_hdf5(url2)) / 255

    img1[img1 > threshold] = 2
    img1[img1 <= threshold] = 0
    img2[img2 > threshold] = 1
    img2[img2 <= threshold] = 0

    img = img_gt + img1 + img2
    print(np.max(img))

    for i in range(img.shape[0]):
        temp = img[i][0]
        get_image2(temp, url_save + "/pred_" + str(i) + ".png")


def draw_detail(base, prop, save_url, save_url_2):
    # print(base.shape, np.average(base), np.average(prop))
    temp = np.zeros(shape=base.shape)
    for i in range(base.shape[0]):
        for j in range(base.shape[1]):
            if prop[i][j] > base[i][j]:
                temp[i][j] = prop[i][j] - base[i][j]
                # temp[i][j]=127
                # if prop[i][j] < base[i][j]:
                #     temp[i][j] = base[i][j] - prop[i][j]
                # temp[i][j] = 127

    #
    cv2.imwrite(save_url, temp)
    temp[temp > 50] = 255
    cv2.imwrite(save_url_2, temp)


def draw_details(base_url, prop_url, num_of_img):
    temp = base_url.split("/")
    save_path = temp[0] + "/" + temp[1] + "/" + temp[2] + "/minus/"
    save_path_2 = temp[0] + "/" + temp[1] + "/" + temp[2] + "/minus_2/"

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_path_2):
        os.mkdir(save_path_2)

    for i in range(num_of_img):
        base = base_url + "shenqi_Prediction_" + str(i) + ".png"
        prop = prop_url + "shenqi_Prediction_" + str(i) + ".png"
        print(base)

        base = cv2.imread(base)
        prop = cv2.imread(prop)
        # print(base.shape, prop.shape)

        base = np.transpose(base, (2, 0, 1))
        prop = np.transpose(prop, (2, 0, 1))

        base = base[0]
        prop = prop[0]

        save_url = save_path + "shenqi_Prediction_" + str(i) + ".png"
        save_url_2 = save_path_2 + "shenqi_Prediction_" + str(i) + ".png"
        print(base.shape, prop.shape)
        draw_detail(base, prop, save_url, save_url_2)


def get_otsu_threshold(url):
    z = 2.3264

    data = load_hdf5(url) / 255
    print(np.shape(data), np.max(data), np.mean(data))

    lthres = filters.threshold_otsu(data)
    uthres = data[data > lthres].mean() + (z * data[data > lthres].std())
    print(lthres, uthres)
    return lthres


if __name__ == '__main__':
    # dataset = ['DRIVE', 'HRF', 'IOSTAR', 'CHASEDB1', 'STARE']
    # net = ['unet3', 'br']
    # threshold = [[0.41568627450980394, 0.376470588235],
    #              [0.470588235294, 0.474509803922],
    #              [0.4, 0.43137254902],
    #              [0.494117647059, 0.466666666667],
    #              [0.447058823529, 0.509803921569]]
    # draw("HRF", "base", threshold=0.5)
    # draw("CHASEDB1", "prop")
    # contrast("CHASEDB1", "base", "prop")

    # for x in range(len(dataset)):
    #     for y in range(len(net)):
    #         draw(dataset[x], net[y])

    dataset = 'HRF'
    num_of_img = 15
    base_url = "./prop_diff/" + dataset + "/base/"
    prop_url = "./prop_diff/" + dataset + "/propose/"
    draw_details(base_url, prop_url, num_of_img)
