from PIL import Image
import numpy as np
import os

public_name = "shenqi_Original_GroundTruth_Prediction"


def joint_img(url, start_number, height_n, width_n):
    originals = np.zeros(shape=(2336, 3504))
    groundtruths = np.zeros(shape=(2336, 3504))
    predictions = np.zeros(shape=(2336, 3504))
    for j in range(height_n):
        for i in range(width_n):
            serial = start_number * height_n * width_n + j * width_n + i
            # print(start_number * height_n * width_n , j * width_n , i)
            img = Image.open(url + public_name + str(serial) + ".png")
            img = np.array(img)

            height = int(img.shape[0] / 3)
            width = img.shape[1]

            original = img[0:height, :]
            groundtruth = img[height:height * 2, :]
            prediction = img[height * 2:, :]

            originals[j * height:(j + 1) * height, i * width:(i + 1) * width] = original
            groundtruths[j * height:(j + 1) * height, i * width:(i + 1) * width] = groundtruth
            predictions[j * height:(j + 1) * height, i * width:(i + 1) * width] = prediction

    pred_img = Image.fromarray(predictions)
    pred_img = pred_img.convert("RGB")

    path = "./result/predictions"
    if not os.path.exists(path):
        os.mkdir(path)
    save_url = path + "/" + url[-2]
    if not os.path.exists(save_url):
        os.mkdir(save_url)

    if start_number % 3 == 0:
        pred_img.save(save_url + "/" + str(int(11 + start_number / 3)) + "_dr.jpg")
    elif start_number % 3 == 1:
        pred_img.save(save_url + "/" + str(int(11 + (start_number - 1) / 3)) + "_g.jpg")
    else:
        pred_img.save(save_url + "/" + str(int(11 + (start_number - 2) / 3)) + "_h.jpg")

    # print(str(start_number + 1) + " pictures have been dealt.")


if __name__ == '__main__':
    height_n = 4
    width_n = 8
    pic_number = 15
    path_number = 1

    for i in range(path_number):
        print("The "+str(i+1)+" round")
        url = "./result/" + str(i) + "/"
        for j in range(pic_number):
            joint_img(url, j, height_n, width_n)

    print("end")
