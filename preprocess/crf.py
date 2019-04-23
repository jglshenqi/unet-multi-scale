import numpy as np
import pydensecrf.densecrf as dcrf
from PIL import Image as Image
from sklearn.metrics import roc_auc_score, roc_curve
import time
import cv2
from matplotlib import pyplot as plt
from pydensecrf.utils import create_pairwise_bilateral


def dense_crf(img, output_probs, inf, para):
    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    # output_probs = output_probs+1
    print(output_probs.max(), output_probs.min())
    output_probs = np.append(1 - output_probs, output_probs, axis=0)
    print(output_probs.max(), output_probs.min())

    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs)
    U = U.reshape((2, -1))
    U = U.astype(np.float32)
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U)

    # d.addPairwiseGaussian(sxy=20, compat=3)
    # d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)
    #
    # d.addPairwiseGaussian(sxy=3, compat=3)
    # d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)
    #
    # pairwise_energy = create_pairwise_bilateral(sdims=(10, 10), schan=(0.01,), img=img, chdim=-1)
    # d.addPairwiseEnergy(pairwise_energy, compat=10)

    sdims = (para[0], para[1])
    schan = (para[2],)
    compat = para[3]
    pairwise_energy = create_pairwise_bilateral(sdims=sdims, schan=schan, img=img, chdim=-1)
    d.addPairwiseEnergy(pairwise_energy, compat=compat)

    Q = d.inference(inf)
    Q = np.array(Q)
    # Q = Q[1]
    print(Q.shape)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))
    # Q = Q.reshape((h, w))

    return Q


def deal(img, pred):
    img = dense_crf(img, pred)
    print(np.shape(img))
    return img


def pred_only_FOV(data_imgs, data_gtruth, test_mask):
    assert (len(data_imgs.shape) == 4 and len(data_gtruth.shape) == 4)  # 4D arrays
    assert (data_imgs.shape[0] == data_gtruth.shape[0])
    assert (data_imgs.shape[2] == data_gtruth.shape[2])
    assert (data_imgs.shape[3] == data_gtruth.shape[3])
    assert (data_imgs.shape[1] == 1 and data_gtruth.shape[1] == 1)  # check the channel is 1
    height = data_imgs.shape[2]
    width = data_imgs.shape[3]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):  # loop over the full images
        for x in range(width):
            for y in range(height):
                if inside_FOV_DRIVE(i, x, y, test_mask) == True:
                    new_pred_imgs.append(data_imgs[i, :, y, x])
                    new_pred_masks.append(data_gtruth[i, :, y, x])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    return new_pred_imgs, new_pred_masks


def inside_FOV_DRIVE(i, x, y, DRIVE_masks):
    assert (len(DRIVE_masks.shape) == 4)  # 4D arrays
    assert (DRIVE_masks.shape[1] == 1)  # DRIVE masks is black and white
    # DRIVE_masks = DRIVE_masks/255.  #NOOO!! otherwise with float numbers takes forever!!

    if x >= DRIVE_masks.shape[3] or y >= DRIVE_masks.shape[2]:  # my image bigger than the original
        return False

    if DRIVE_masks[i, 0, y, x] > 0:  # 0==black pixels
        # print DRIVE_masks[i,0,y,x]  #verify it is working right
        return True
    else:
        return False


def print_roc(img):
    # url = './shenqi_Prediction_0.png'
    # img = Image.open(url)
    # img = np.array(img) / 255
    # print(img.shape, img.max(), img.min())

    gt = Image.open("./img_original/gt.png")
    gt = np.array(gt)
    # print(gt.shape, gt.max(), gt.min())

    mask = Image.open("./img_original/mask.jpg")
    mask = np.array(mask) / 255
    # print(mask.shape, mask.max(), mask.min())

    img = img.reshape((1, 1, 960, 999))
    gt = gt.reshape((1, 1, 960, 999))
    mask = mask.reshape((1, 1, 960, 999))
    pred, true = pred_only_FOV(img, gt, mask)
    # print(true.shape, pred.shape)
    fpr, tpr, tresholds = roc_curve(true, pred)
    # print(tresholds)
    # stop = input()
    roc = roc_auc_score(true, pred)
    print('the number of roc is:', roc)

    fpr, tpr, thresholds = roc_curve(true, pred)
    AUC_ROC = roc_auc_score(true, pred)
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
    print("\nArea under the ROC curve: " + str(AUC_ROC))
    print(fpr.max(), fpr.min(), tpr.max(), tpr.min())
    stop = input()
    # roc_curve = plt.figure()
    plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    # plt.scatter(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.scatter(fpr, tpr)
    plt.title('ROC curv3e')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig("./ROC.png")
    # stop =input()


def dataset_normalized(imgs):
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    new_imgs = cv2.LUT(np.array(imgs, dtype=np.uint8), table)
    return new_imgs


def clahe_equalized(imgs):
    # imgs_equalized = np.empty(imgs.shape)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = clahe.apply(np.array(imgs, dtype=np.uint8))
    return imgs_equalized


def rgb2gray(rgb):
    bn_imgs = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    bn_imgs = np.reshape(bn_imgs, (rgb.shape[0], rgb.shape[1]))
    return bn_imgs


def my_PreProc(data):
    train_imgs = rgb2gray(data)
    # cv2.imwrite("./temp.png",train_imgs)
    train_imgs = dataset_normalized(train_imgs)
    # cv2.imwrite("./temp2.png", train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    # cv2.imwrite("./temp3.png", train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    # cv2.imwrite("./temp4.png", train_imgs)
    return train_imgs


if __name__ == '__main__':
    url_p = './img_original/shenqi_Prediction_0.png'
    url_o = "./img_original/Image_11L.jpg"
    #
    # url_o = "./img_original/im1.png"
    # url_p = "./img_original/anno1.png"
    pred = Image.open(url_p)
    pred = np.array(pred)
    print(pred.shape, pred.max(), pred.min(), np.unique(pred))

    img = Image.open(url_o)
    img = np.array(img)
    img = my_PreProc(img)
    # img = np.expand_dims(img, axis=-1)
    print(img.shape, img.max(), img.min(), np.unique(img))

    for i in range(15):
        # para = [10, 10, 0.01, 10]
        ex = (i+1)*3
        para = [ex, 10, 0.01, 1.0]
        start_time = time.time()
        inf = 50
        temp = dense_crf(img, pred, inf, para)

        temp = np.array(temp) * 255
        print(temp.shape, temp.max(), temp.min(), np.unique(temp))

        cv2.imwrite("./img_result/" + str(ex) + ".png", temp)
        print(str(i) + "ends,it cost", str(int(time.time() - start_time)), "seconds.")
