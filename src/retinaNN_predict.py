# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from keras.models import model_from_json
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, \
    jaccard_similarity_score, f1_score
from skimage import filters
import sys
import time
from extract_patches import recompone_image, recompone_overlap, get_data_testing_overlap, pred_only_FOV
from help_functions import get_model, visualize, load_hdf5, prepare_data
import get_config
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

sys.path.append("./src/network/")
plt.switch_backend('agg')
sys.path.insert(0, './lib/')

config = get_config.get_config()
config.choose_GPU()

path_experiment = './' + config.name_experiment + '/'
path_save = path_experiment + 'result' + '/'
if not os.path.exists(path_save):
    os.mkdir(path_save)


def visualized(pred_patches,
               test_mask,
               new_height,
               new_width,
               stride_height,
               stride_width,
               counter,
               patch_height,
               patch_width,
               new_path_experiment):
    # ========== Elaborate and visualize the predicted images ====================
    full_img_height = np.shape(test_mask)[2]
    full_img_width = np.shape(test_mask)[3]
    # turn pred patches into images
    if config.training_format == 0:
        pred_imgs = recompone_image(pred_patches, full_img_height, full_img_width, patch_height, patch_width)
    elif config.training_format == 1:
        pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)
    else:
        pred_imgs = pred_patches
    print("pred imgs shape:", pred_imgs.shape, "range:", pred_imgs.min(), "-", pred_imgs.max(),
          "average is:", pred_imgs.mean())

    pred_imgs = pred_imgs[:, :, 0:full_img_height, 0:full_img_width]
    # if it os necessary for showing the image in a threshold

    if config.vi_threshold:
        print("\nvisualized:  Costum threshold (for positive) of " + str(config.vi_threshold))
        for i in range(pred_imgs.shape[2]):
            for j in range(pred_imgs.shape[3]):
                pred_imgs[pred_imgs >= config.vi_threshold] = 1
                pred_imgs[pred_imgs < config.vi_threshold] = 1

    pred_imgs = pred_imgs[0]
    pred_imgs = np.transpose(pred_imgs, (1, 2, 0))
    # save the pred images
    visualize(pred_imgs, new_path_experiment + config.name_experiment + "_Prediction_" + str(counter))


def get_pred_patches(patches_imgs_test):
    # ================ Run the prediction of the patches ==================================
    best_last = config.best_last
    try:
        model = model_from_json(open(path_experiment + config.name_experiment + '_architecture.json').read())
    except:
        shape_input = np.shape(patches_imgs_test)
        print(shape_input)
        save_url = './' + config.name_experiment + '/' + config.name_experiment + '_architecture.json'
        model = get_model(shape_input, config, save_url)

    # model.load_weights(path_experiment + "shenqi_00_weights.h5")
    model.load_weights(path_experiment + config.name_experiment + '_' + best_last + '_weights.h5')
    # Calculate the predictions
    pred_patches = model.predict(patches_imgs_test, batch_size=config.batch_size, verbose=2)
    pred_patches = np.array(pred_patches)

    pred_patches = np.array(pred_patches)
    if len(pred_patches.shape) == 4:
        pps = pred_patches.shape
        pred_patches = pred_patches.reshape((1, pps[0], pps[1], pps[2], pps[3]))

    if config.training_format == 0 and len(pred_patches.shape) == 2:
        pps = pred_patches.shape
        pred_patches = pred_patches.reshape((1, pps[0], pps[1]))
        if config.softmax_index == 1:
            pred_patches = pred_patches[:, :, 1:2]
    elif config.softmax_index == 1:
        pred_patches = pred_patches[:, :, 1:2, :, :]

    return pred_patches


def predict_test():
    for i in range(0, config.full_images_to_test):
        now = time.time()
        print("\n===============the %d/%d round===============" % (i, config.full_images_to_test))
        # get the testing patches
        patches_imgs_test, test_mask, new_height, new_width = get_data_testing_overlap(
            DRIVE_test_imgs_original=config.test_imgs_original,
            DRIVE_test_mask=config.test_mask,
            start_img=i,
            patch_height=config.patch_height,
            patch_width=config.patch_width,
            stride_height=config.stride_height,
            stride_width=config.stride_width,
            training_format=config.training_format)
        # predict rhe testing patches
        print("patches_imgs_test.shape", patches_imgs_test.shape, "range:", patches_imgs_test.min(), "-",
              patches_imgs_test.max(), "average is:", patches_imgs_test.mean())
        predicision = get_pred_patches(patches_imgs_test)
        print("predictions shape", predicision.shape, "range:", predicision.min(), "-", predicision.max(),
              "average is:", predicision.mean())

        # turn the predictions into images
        for j in range(config.num_of_loss):
            number = j
            if not os.path.exists(path_experiment + str(number)):
                os.mkdir(path_experiment + str(number))
            if config.num_of_loss == 1:
                pred_patches = predicision[0]
            else:
                pred_patches = predicision[j]

            new_path_experiment = path_experiment + str(number) + "/"

            visualized(pred_patches=pred_patches,
                       test_mask=test_mask,
                       new_height=new_height,
                       new_width=new_width,
                       stride_height=config.stride_height,
                       stride_width=config.stride_width,
                       counter=i,
                       patch_height=config.patch_height,
                       patch_width=config.patch_width,
                       new_path_experiment=new_path_experiment)

        print("===time: ", time.time() - now, "===")


def predict_train():
    for i in range(0, config.full_images_to_train):
        print("\n===============the %d/%d round===============" % (i, config.full_images_to_train))
        # get the testing patches
        patches_imgs_test, test_mask, new_height, new_width = get_data_testing_overlap(
            DRIVE_test_imgs_original=config.train_imgs_original,
            DRIVE_test_mask=config.train_mask,
            start_img=i,
            patch_height=config.patch_height,
            patch_width=config.patch_width,
            stride_height=config.stride_height,
            stride_width=config.stride_width,
            training_format=config.training_format)
        # predict rhe testing patches
        predicision = get_pred_patches(patches_imgs_test)

        # turn the predictions into images
        if not os.path.exists(path_experiment + "threshold"):
            os.mkdir(path_experiment + "threshold")
        if config.num_of_loss == 1:
            pred_patches = predicision[0]
        else:
            pred_patches = predicision[config.num_of_loss - 1]

        new_path_experiment = path_experiment + "train_images" + "/"

        visualized(pred_patches=pred_patches,
                   test_mask=test_mask,
                   new_height=new_height,
                   new_width=new_width,
                   stride_height=config.stride_height,
                   stride_width=config.stride_width,
                   counter=i,
                   patch_height=config.patch_height,
                   patch_width=config.patch_width,
                   new_path_experiment=new_path_experiment)


def get_otsu_threshold(url):
    data = load_hdf5(url) / 255
    lthres = filters.threshold_otsu(data)
    return lthres


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


def evalu_roc(pred_url, save_url, threshold=0.5):
    y_true = load_hdf5(config.test_groundtruth) / 255
    y_pred = load_hdf5(pred_url) / 255
    mask = load_hdf5(config.test_mask) / 255
    print(y_true.shape)
    print(y_true.max())
    print("groundtruth:", y_true.shape, "range:", str(y_true.min()), "-" + str(y_true.max()))
    print("prediction:", y_pred.shape, "range:", str(y_pred.min()), "-" + str(y_pred.max()))
    print("mask:", mask.shape, "range:", str(mask.min()), "-" + str(mask.max()))

    score, true = pred_only_FOV(y_pred, y_true, mask)
    print(true.max(), true.min(), score.max(), score.min())

    evaluate(true, score, save_url, threshold)


def get_predict():
    start_time = time.time()
    # get the pred images and save
    print("=======predict the images=======")
    predict_test()
    print("Prediction ended,it costs", str(int(time.time() - start_time)) + " seconds")
    start_time = time.time()

    # turn the pred images into hdf5 files,just for last output
    print("=======transfer images to hdf5=======")
    url_original = [path_experiment + str(config.num_of_loss - 1) + "/"]
    url_save = [path_save + config.dataset + "_" + config.get_net + ".hdf5"]
    prepare_data(url_original, url_save)
    print("Transfering ended,it costs", str(int(time.time() - start_time)) + " seconds")
    start_time = time.time()

    # calculate the threshold of best accuancy in training images
    print("=======calcculate the threshold=======")
    # threshold = get_threshold()
    threshold = get_otsu_threshold(url_save[0])
    # threshold = 0.5
    print("calcculate the threshold ended,it costs", str(int(time.time() - start_time)) + " seconds")
    start_time = time.time()

    # evaluate the results from hdf5 files
    print("=======evaluate=======")
    pred_url = path_save + config.dataset + "_" + config.get_net + ".hdf5"
    evalu_roc(pred_url, path_save, threshold)
    if os.path.exists(pred_url):
        os.remove(pred_url)
    print("Evaluation ended,it costs", str(int(time.time() - start_time)) + " seconds")


if __name__ == '__main__':
    get_predict()


def cal_threshold(y_true, y_scores):
    thresholds = np.arange(0, 255, 1) / 255

    acc = 0
    index = 0
    print("")
    for i in range(np.shape(thresholds)[0]):
        y_pred = np.array(y_scores)

        y_pred[y_pred >= thresholds[i]] = 1
        y_pred[y_pred < thresholds[i]] = 0

        confusion = confusion_matrix(y_true, y_pred)

        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))

        if accuracy >= acc:
            acc = accuracy
            index = i
        # print("calculate now:", i, thresholds[i], accuracy, "max history:", index, thresholds[index], acc, end='\r')
        print("calculate now:", i, thresholds[i], accuracy, "max history:", index, thresholds[index], acc)
        # else:
        #     break

    print("max history:", index, thresholds[index], acc)
    print("")
    return thresholds[index]


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
