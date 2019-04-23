import h5py
import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


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


def extract_random_in(train_mask, patch_h, patch_w, N_patches):
    print("mask shape is:", train_mask.shape)

    patch_per_img = int(N_patches / train_mask.shape[0])  # N_patches equally divided in the full images
    iter_tot = 0  # iter over the total numbe rof patches (N_patches)
    list1 = []
    list_2 = []
    one = 0
    zero = 0

    print(zero, one)
    print(np.shape(list1))
    return list1


def draw_center():
    url_ori = "../dataset/HRF/HRF_datasets_training_testing"

    train_imgs = load_hdf5(url_ori + "/" + "HRF_dataset_groundTruth_train.hdf5") / 255 * 2
    train_results = load_hdf5('./draw_center/temp.hdf5') / 255

    train_results[train_results >= 0.5] = 1
    train_results[train_results < 0.5] = 0

    train_masks = train_imgs + train_results

    train_img = np.array(train_imgs[0][0]) * 255
    train_result = np.array(train_results[0][0]) * 255
    train_mask = np.array(train_masks[0][0]) / 3 * 255

    # Image.fromarray(train_img).show()
    # Image.fromarray(train_result).show()
    # Image.fromarray(train_mask).show()
    # stop = input()
    # train_nodes = load_node(url_ori + "/train_patch.pickle")

    print(np.shape(train_imgs), np.shape(train_results))
    patch_w = 48
    patch_h = 48
    list_all = []

    for i in range(10, 12):  # loop over the full images
        print("No.", str(i))
        k = 0
        list_ = []

        mul = 1
        train_mask = train_masks[i][0]
        cal_b = 0
        cal_w = 0
        cal_g = 0
        cal_num = 0
        while (1):
            # cal_num += 1
            # if cal_num % 100 == 0:
            #     stop = input()
            if len(list_) == 8000:
                break
            x_center = np.random.randint(0 + int(patch_w / 2), (train_masks.shape[2] - int(patch_w / 2)) / mul)
            y_center = np.random.randint(0 + int(patch_h / 2), int((train_masks.shape[3] - int(patch_h / 2)) / mul))
            # x_center = np.random.randint(0 + int(patch_w / 2), (train_mask.shape[3] - int(patch_w / 2)) / mul,
            #                              patch_per_img)
            # y_center = np.random.randint(0 + int(patch_h / 2), int((train_mask.shape[2] - int(patch_h / 2)) / mul),
            #                              patch_per_img)
            patch = train_mask[x_center - 24:x_center + 24, y_center - 24:y_center + 24]

            # img = np.array(train_mask) / 3 * 255
            # img = Image.fromarray(np.array(img))
            # img.show()
            # stop = input()

            temp = [np.sum(patch == 0), np.sum(patch == 1), np.sum(patch == 2), np.sum(patch == 3)]
            temp = np.array(temp)
            # print(temp, end=" ")
            #
            # if np.sum(temp) == 0:
            #     print(x_center, y_center, patch, train_mask[x_center - 24:x_center + 24, y_center - 24:y_center + 24])
            #     stop = input()

            if [x_center, y_center] not in list_:

                if (temp[0] + temp[2]) / np.sum(temp) == 1 and cal_b < 1600:
                    cal_b += 1
                    list_.append([x_center, y_center])
                    print("get a black center:", str(i), "  ", cal_b)
                elif temp[1] > 230 and cal_g < 1600:
                    cal_g += 1
                    list_.append([x_center, y_center])
                    print("get a green center:", str(i), "  ", cal_g)
                elif temp[1] + temp[3] > 0 and cal_w < 4800:
                    cal_w += 1
                    list_.append([x_center, y_center])
                    print("get a white center:", str(i), "  ", cal_w)
                else:
                    continue

        list_all.append(list_)
        save_node(list_, './draw_center/' + str(i) + ".pickle")

        # save_node(list_all, "./draw_center.pickle")


def test_center():
    url_ori = "../dataset/HRF/HRF_datasets_training_testing"

    train_imgs = load_hdf5(url_ori + "/" + "HRF_dataset_groundTruth_train.hdf5") / 255 * 2
    train_results = load_hdf5('./draw_center/temp.hdf5') / 255

    train_results[train_results >= 0.5] = 1
    train_results[train_results < 0.5] = 0

    train_imgs = train_imgs + train_results

    # train_nodes = load_node("./draw_center/" + "/0.pickle")

    # print(np.shape(train_imgs), np.shape(train_results), np.shape(train_nodes))

    cal = np.zeros(shape=(30, 4))

    cal2 = [0, 0, 0, 0]
    num2 = 0

    for i in range(np.shape(train_imgs)[0]):
        num = 0
        num3 = 0
        num4 = 0

        train_img = train_imgs[i][0]
        # train_result = train_results[i][0]
        # train_node = load_node("./draw_center/" + "/" + str(i) + ".pickle")
        train_node = load_node("./draw_center/train_patch48_240000")

        for node in train_node[0]:
            w = node[1]
            h = node[0]

            patch = train_img[w - 24:w + 24, h - 24:h + 24]
            patch = np.array(patch)

            # cal[i][0] += np.sum(patch == 0)
            # cal[i][1] += np.sum(patch == 1)
            # cal[i][2] += np.sum(patch == 2)
            # cal[i][3] += np.sum(patch == 3)

            temp = [np.sum(patch == 0), np.sum(patch == 1), np.sum(patch == 2), np.sum(patch == 3)]

            cal[i] += temp
            if (temp[0] + temp[2]) / np.sum(temp) == 1:
                num += 1
                num2 += 1
            elif temp[1] >= 230:
                num3 = num3 + 1
            elif temp[1] + temp[3] > 0:
                num4 += 1
                # print(np.array(temp) / 2304)
        #
        print(num)
        print(num3)
        print(num4)

        print(cal[i] / np.sum(cal[i]) * 100)

        cal2[0] += np.sum(train_img == 0)
        cal2[1] += np.sum(train_img == 1)
        cal2[2] += np.sum(train_img == 2)
        cal2[3] += np.sum(train_img == 3)

        a = np.sum(cal[:, 0:1])
        b = np.sum(cal[:, 1:2])
        c = np.sum(cal[:, 2:3])
        d = np.sum(cal[:, 3:4])

        k = a + b + c + d
        print(a / k, b / k, c / k, d / k)

        print(np.array(cal2) / sum(cal2) * 100)

        stop = input()
    print(num2)


def get_together():
    train_nodes = []
    for i in range(30):
        train_node = load_node("./draw_center/" + str(i) + ".pickle")
        # for node in train_node:
        #     temp = node[0]
        #     node[0] = node[1]
        #     node[1] = temp
        # save_node(train_node, "./draw_center/" + str(i) + ".pickle")
        train_nodes.append(train_node)
    print(np.shape(train_nodes))
    save_node(train_nodes, "./draw_center/train_patch48_240000")



if __name__ == '__main__':
    test_center()
    # draw_distribution()
    # draw_center()
    # adjust_center()
    # get_together()
