# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from keras.models import model_from_json
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from skimage import filters
import sys
import time
from PIL import Image
import numpy as np
import h5py
from matplotlib import pyplot as plt
import scipy.misc as misc
from skimage import io

# model name
path_experiment = './NFN/'
path_save = path_experiment + 'result' + '/'
if not os.path.exists(path_save):
    os.mkdir(path_save)


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["image"][()]


# write images into hdf5 files
def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


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


def get_datasets(imgs_dir):
    print(imgs_dir)
    imgs = None
    for path, subdirs, files in os.walk(imgs_dir):

        img = Image.open(path + files[0])
        size = np.shape(img)
        print("the shape of images is:", np.shape(img))

        if np.array(size).shape[0] == 2:
            size = [size[0], size[1], 1]

        imgs = np.zeros((len(files), size[0], size[1], size[2]))

        print("")
        for i in range(len(files)):
            file = "shenqi_Prediction_" + str(i) + ".png"
            # file = files[i]

            print(imgs_dir + file)
            img = Image.open(imgs_dir + file)
            img = np.array(img)

            if np.shape(img.shape)[0] == 2:
                img = img.reshape((img.shape[0], img.shape[1], 1))
            imgs[i] = img

        imgs = np.transpose(imgs, (0, 3, 1, 2))
        # show_img(imgs)
        print("")

    return imgs


def prepare_data(urls_original, urls_save):
    for i in range(urls_original.__len__()):
        img = get_datasets(urls_original[i])
        write_hdf5(img, urls_save[i])


def evaluate_roc(y_true, y_scores, save_url):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
    print("\nArea under the ROC curve: " + str(AUC_ROC))
    # roc_curve = plt.figure()
    plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(save_url + "ROC.png")

    return AUC_ROC


def evaluate_prc(y_true, y_scores, save_url):
    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))
    prec_rec_curve = plt.figure()
    plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig(save_url + "Precision_recall.png")

    return AUC_prec_rec


def confusion_matrixd(y_true, y_scores, threshold_confusion=0.5):
    # Confusion matrix
    y_pred = np.array(y_scores)
    # threshold_confusion = 0.5
    print("\nConfusion matrix:  Costum threshold (for positive) of " + str(threshold_confusion))

    y_pred[y_pred >= threshold_confusion] = 1
    y_pred[y_pred < threshold_confusion] = 0

    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    print("Global Accuracy: " + str(accuracy))
    specificity = 0
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    print("Specificity: " + str(specificity))
    sensitivity = 0
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    print("Sensitivity: " + str(sensitivity))
    precision = 0
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    print("Precision: " + str(precision))

    return confusion, y_pred, accuracy, specificity, sensitivity, precision


def jacs(y_true, y_pred):
    # Jaccard similarity index
    jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
    print("\nJaccard similarity score: " + str(jaccard_index))

    return jaccard_index


def F1_s(y_true, y_pred):
    # F1 score
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    print("\nF1 score (F-measure): " + str(F1_score))

    return F1_score


def evaluate(y_true, y_scores, save_url, threshold=0.5):
    AUC_ROC = evaluate_roc(y_true, y_scores, save_url)
    AUC_prec_rec = evaluate_prc(y_true, y_scores, save_url)
    # AUC_ROC = 0
    # AUC_prec_rec=0
    confusion, y_pred, accuracy, specificity, sensitivity, precision = confusion_matrixd(y_true, y_scores, threshold)
    jaccard_index = jacs(y_true, y_pred)
    F1_score = F1_s(y_true, y_pred)

    # Save the results
    file_perf = open(save_url + 'performance.txt', 'w')
    file_perf.write("Area under the ROC curve: " + str(AUC_ROC)
                    + "\r\nArea under Precision-Recall curve: " + str(AUC_prec_rec)
                    + "\r\nJaccard similarity score: " + str(jaccard_index)
                    + "\r\nF1 score (F-measure): " + str(F1_score)
                    + "\r\n\r\nConfusion matrix:  Costum threshold (for positive) of " + str(threshold)
                    + "\r\nConfusion matrix:"
                    + str(confusion)
                    + "\r\nACCURACY: " + str(accuracy)
                    + "\r\nSENSITIVITY: " + str(sensitivity)
                    + "\r\nSPECIFICITY: " + str(specificity)
                    + "\r\nPRECISION: " + str(precision)
                    )
    file_perf.close()


def show_img(img):
    imgs = []
    for i in range(img.shape[0]):
        temp = img[i][0]
        imgs.extend(temp)
    print(np.shape(imgs))
    Image.fromarray(np.array(imgs)).show()


def evalu_roc(dataset, pred_url, save_url, threshold=0.5):
    gtruth_url = "../dataset/DRIVE/DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_test.hdf5".replace('DRIVE',
                                                                                                                dataset)
    mask_url = "../dataset/DRIVE/DRIVE_datasets_training_testing/DRIVE_dataset_borderMasks_test.hdf5".replace('DRIVE',
                                                                                                              dataset)

    y_true = load_hdf5(gtruth_url) / 255
    y_pred = load_hdf5(pred_url) / 255
    mask = load_hdf5(mask_url) / 255
    print("test", y_true.max(), y_pred.max(), mask.max(), y_true.mean(), y_pred.mean(), mask.mean(), y_true.min(),
          y_pred.min(), mask.min())
    print(y_true.shape, y_pred.shape, mask.shape)

    score, true = pred_only_FOV(y_pred, y_true, mask)
    print(true.max(), true.min())
    evaluate(true, score, save_url, threshold)


def get_otsu_threshold(url):
    data = load_hdf5(url) / 255

    lthres = filters.threshold_otsu(data)
    return lthres


def combine(url1, url2, url_combine):
    url1 = url1[0]
    url2 = url2[0]
    url_combine = url_combine[0]
    if not os.path.exists(url_combine):
        os.mkdir(url_combine)
    for path, subdirs, files in os.walk(url1):
        for i in range(len(files)):
            file = "shenqi_Prediction_" + str(i) + ".png"
            # file = files[i]

            print(file)
            # img1 = Image.open(url1 + file)
            img1 = misc.imread(url1 + file)
            img1 = np.array(img1)
            print(img1.shape, img1.max(), img1.min(), img1.mean())

            # img2 = Image.open(url2 + file)
            img2 = misc.imread(url2 + file)
            img2 = np.array(img2)
            print(img2.shape, img2.max(), img2.min(), img2.mean())

            img = img1 / 2 + img2 / 2
            img = img.astype(int)
            print(img.shape, img.max(), img.min(), img.mean())

            # for x in [111, 123, 127, 138, 332, 238]:
            #     for y in [123, 456, 368, 356]:
            #         print(img[x][y], img1[x][y], img2[x][y])
            # img = img.transpose((1,0))
            io.imsave(url_combine + file,img)
            # img = Image.fromarray(img)
            # # img.show()
            # img.save(url_combine + file, format="PNG")

            # stop = input()


def get_predict():
    dataset = "CHASEDB1"
    get_net = "NFN"
    start_time = time.time()

    # turn the pred images into hdf5 files,just for last output
    url1 = [path_experiment + "3/1/"]
    url2 = [path_experiment + "4/1/"]
    url_combine = [path_experiment + "combine/"]
    combine(url1, url2, url_combine)
    print("=======transfer images to hdf5=======")

    # url_original = ["./shenqi/test/"]
    # url_original = [temp]
    url_save = [path_save + dataset + "_" + get_net + ".hdf5"]
    prepare_data(url_combine, url_save)
    print("Transfering ended,it costs", str(int(time.time() - start_time)) + " seconds")
    start_time = time.time()

    # calculate the threshold of best accuancy in training images
    print("=======calcculate the threshold=======")
    # threshold = get_threshold()
    print(url_save)
    threshold = get_otsu_threshold(url_save[0])
    # threshold = 0.5
    print("calcculate the threshold ended,it costs", str(int(time.time() - start_time)) + " seconds")
    start_time = time.time()

    # evaluate the results from hdf5 files
    print("=======evaluate=======")
    pred_url = path_save + dataset + "_" + get_net + ".hdf5"
    evalu_roc(dataset, pred_url, path_save, threshold)
    if os.path.exists(pred_url):
        os.remove(pred_url)
    print("Evaluation ended,it costs", str(int(time.time() - start_time)) + " seconds")


def draw_ROC():
    gtruth_url = "./dataset/DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_test.hdf5"
    mask_url = "./dataset/DRIVE_datasets_training_testing/DRIVE_dataset_borderMasks_test.hdf5"

    y_true = load_hdf5(gtruth_url) / 255
    mask = load_hdf5(mask_url) / 255

    dataset = "DRIVE"
    pred_net = ["unet3", "N4"]
    for i in range(len(pred_net)):
        pred_url = "./shenqi/" + dataset + "_" + pred_net[i] + ".hdf5"
        y_pred = load_hdf5(pred_url)
        print(y_true.shape, y_pred.shape)
        y_pred, true = pred_only_FOV(y_pred, y_true, mask)
        print(true.shape, y_pred.shape)
        fpr, tpr, thresholds = roc_curve(true, y_pred)
        AUC_ROC = roc_auc_score(true, y_pred)
        plt.plot(fpr, tpr, '-', label='Area Under the ' + pred_net[i] + ' Curve (AUC = %0.4f)' % AUC_ROC)

    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig("./shenqi/ROC.png")


if __name__ == '__main__':
    get_predict()
