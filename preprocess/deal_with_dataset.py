import os
from help_functions import *
import scipy.misc as misc
from PIL import Image as Image


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["image"][()]


def read(url):
    # 读取一个路径下所有子文件和子文件夹下的文件
    # 输入为根目录，返回值为每一个文件的目录和文件名
    path = []
    name = []

    for dirpath, dirnames, filenames in os.walk(url):
        for i in range(filenames.__len__()):
            path.append(dirpath)
            name.append(filenames[i])

    return path, name


def deal_dataset(url):
    data = load_hdf5(url)
    print(url, data.max())
    if (data.max() == 1):
        # data = data * 255
        print("stop!")
        stop = input()
        # write_hdf5(data, url)


def max_255():
    path, name = read("../dataset")
    print(path)
    print(name)
    for i in range(len(path)):
        url = path[i] + "/" + name[i]
        if "hdf5" in url:
            deal_dataset(url)


def grey_mask(url):
    path, file = read(url)
    print(path)
    print(file)
    print(len(file))
    for i in range(len(file)):
        img_url = path[i] + "/" + file[i]
        img = misc.imread(img_url)
        print(img_url, end=" ")
        img = np.array(img)
        img = img.transpose((1, 0))
        # a = img[0]
        # b = img[1]
        # c = img[2]
        # b = b - a
        # c = c - a
        # print(img.shape, a.max(), a.min(), b.max(), b.min(), c.max(), c.min())

        img = Image.fromarray(img)
        img.save(img_url.replace("1st_m", "m").replace("gif", "png"))


def resize(url, size):
    path, file = read(url)
    print(path)
    print(file)
    print(len(file))
    for i in range(len(file)):
        img_url = path[i] + "/" + file[i]
        img = misc.imread(img_url)
        print(img_url, img.shape, end="")
        w = size[0]
        h = size[1]
        resize_image = misc.imresize(img, [w, h], interp='nearest')
        print(resize_image.shape)
        im = Image.fromarray(resize_image)
        im.save(img_url)


def cal_mean(url):
    url_train = url + 'train.hdf5'
    imgs = load_hdf5(url_train)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    print("train: mean is " + str(imgs_mean), "std is ", str(imgs_std))

    url_test = url + 'test.hdf5'
    imgs = load_hdf5(url_test)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    print("test: mean is " + str(imgs_mean), "std is ", str(imgs_std))


def for_test(url):
    path, name = read(url)
    for i in range(len(path)):
        img_url = path[i] + "/" + name[i]
        img = misc.imread(img_url)
        img = np.array(img)
        print(img_url, img.shape)
        # img = Image.fromarray(img)
        # img.save(img_url.replace('gif','png'))


def rename(url):
    path, name = read(url)
    for i in range(len(path)):
        img_url = path[i] + "/" + name[i]
        os.rename(img_url, img_url.replace("_manual1", "").replace("shenqi_Prediction_","").replace("_test",""))
        print(img_url)


def for_test2():
    url = '../dataset'
    path, name = read(url)
    for i in range(len(path)):
        img_url = path[i] + '/' + name[i]
        if 'hdf5' in img_url:
            img = load_hdf5(img_url)
            print(img_url, img.shape, img.max(), img.min())


def change_format(url):
    path, name = read(url)

    for i in range(len(path)):
        img_url = path[i] + "/" + name[i]
        if 'png' in img_url:
            continue
        else:
            img = misc.imread(img_url)
            img = Image.fromarray(np.array(img))
            print(img_url, img_url[:-3] + 'png')
            img.save(img_url[:-3] + 'png')
            os.remove(img_url)


if __name__ == '__main__':
    dataset = 'IOSTAR'

    size = [512, 512]
    # url = "./temp/" + dataset
    # url = './temp'
    # resize(url, size)

    # change_format('./temp/' + dataset)

    # url_test = './temp/' + dataset + "/test/mask"
    # url_train = './temp/' + dataset + "/training/mask"
    # grey_mask(url_test)
    # grey_mask(url_train)

    # url = "./temp/" + dataset + "_datasets_training_testing/" + dataset + '_dataset_imgs_'
    # cal_mean(url)

    url = "./temp/"
    rename(url)

    # max_255()
    # for_test("./temp/" + dataset)
    # for_test2()
