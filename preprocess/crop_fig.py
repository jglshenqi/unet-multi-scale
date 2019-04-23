import cv2
import os


def crop_fig(url, start_point, patch_size, save_url):
    gt = cv2.imread(url)
    stride = [patch_size, patch_size]
    crop = gt[start_point[0]:start_point[0] + stride[0], start_point[1]:start_point[1] + stride[1]]
    cv2.imwrite(save_url, crop)


def crop_figs(urls, lists, patch_size):
    names = ["img", "gt", "base", "prop"]

    for i in range(len(urls)):
        for cor in lists:
            save_path = "./prop_diff/" + dataset + "/patch/" + str(cor[0]) + "_" + str(cor[1]) + "_" + str(
                patch_size) + "/"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_url = save_path + names[i] + ".png"
            cor = [cor[1], cor[0]]
            print(save_url)
            crop_fig(urls[i], cor, patch_size, save_url)


if __name__ == '__main__':
    dataset = 'CHASEDB1'

    if dataset == 'CHASEDB1':
        num = 2
        img_url = "./prop_diff/" + dataset + "/img/Image_11L.jpg"
        gt_url = "./prop_diff/" + dataset + "/img/Image_11L_1stHO.png"
        # lists = [[609, 146], [748, 126], [135, 397], [97, 309], [202, 267]]
        # lists = [[609, 146], [135, 309]]
        lists = [[492,378]]
    else:
        num = 9
        img_url = "./prop_diff/" + dataset + "/img/14_dr.png"
        gt_url = "./prop_diff/" + dataset + "/img/14_dr_gt.png"
        # lists = [[350, 680], [350, 1670], [1057, 125], [1178, 1053], [1570, 700]]
        lists = [[250, 580]]
    base_url = "./prop_diff/" + dataset + "/base/" + "shenqi_prediction_" + str(num) + ".png"
    prop_url = "./prop_diff/" + dataset + "/propose/" + "shenqi_prediction_" + str(num) + ".png"

    urls = [img_url, gt_url, base_url, prop_url]
    #
    #
    # lists = [[0,0]]
    patch_size = 100
    crop_figs(urls, lists, patch_size)
