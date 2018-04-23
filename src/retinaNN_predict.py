# -*- coding: utf-8 -*-
import configparser
# Keras
import os
from keras.models import model_from_json
from keras.models import Model
# scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys
import gc

# help_functions.py
from help_functions import *
# extract_patches.py
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import paint_border
from extract_patches import kill_border
from extract_patches import pred_only_FOV
from extract_patches import get_data_testing
from extract_patches import get_data_testing_overlap
# pre_processing.py
from pre_processing import my_PreProc
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

plt.switch_backend('agg')
sys.path.insert(0, './lib/')

# ========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('./configuration.txt')
# ===========================================
# run the training on invariant or local
dataset = config.get('public', 'dataset')
path_data = config.get('data paths', 'path_local').replace("DRIVE", dataset)
# original test images (for FOV selection)
DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original').replace("DRIVE", dataset)
# the border masks provided by the DRIVE
# DRIVE_test_border_masks = path_data + config.get('data paths', 'test_border_masks').replace("DRIVE",dataset)
# test_border_masks = load_hdf5(DRIVE_test_border_masks)
# dimension of the patches
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
# the stride in case output with average
stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))
full_images_to_test = int(config.get(dataset, 'full_images_to_test'))
assert (stride_height < patch_height and stride_width < patch_width)
# model name
name_experiment = config.get('experiment name', 'name')
path_experiment = './' + name_experiment + '/'
# N full images to be predicted
# Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
# Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))
# ====== average mode ===========
net_name = config.get('public', 'network')
average_mode = config.getboolean('testing settings', 'average_mode')
num_of_loss = int(config.get(net_name, 'num_of_loss'))
masks_original = int(config.get(net_name, 'mask_original'))
softmax = int(config.get(net_name, 'softmax'))
part = int(config.get(dataset, "part"))
full_images_to_test = int(full_images_to_test / part)
type_of_output = int(config.get(net_name, 'type_of_output'))
if type_of_output == 1:
    num_of_loss = 1


def get_pred_patches(patches_imgs_test):
    # ================ Run the prediction of the patches ==================================
    best_last = config.get('testing settings', 'best_last')

    model = model_from_json(open(path_experiment + name_experiment + '_architecture.json').read())
    model.load_weights(path_experiment + name_experiment + '_' + best_last + '_weights.h5')
    # Calculate the predictions
    print("patches_imgs_test.shape", patches_imgs_test.shape)
    pred_patches = model.predict(patches_imgs_test, batch_size=4, verbose=2)

    if masks_original == 0:
        pred = pred_to_imgs(pred_patches, patch_height, patch_width, "original")
    elif softmax:
        pred = []
        if type_of_output == 1:
            pred = down_to_imgs(pred_patches[3], "original")
        elif num_of_loss == 1:
            pred = down_to_imgs(pred_patches, "original")
        else:
            for i in range(num_of_loss):
                pred_patch = down_to_imgs(pred_patches[i], "original")
                pred.append(pred_patch)
    else:
        pred = pred_patches

    pred = np.array(pred)

    return pred


def visualized(patches_imgs_test, test_mask, pred_patches, test_groundtruth, new_height, new_width, number, counter):
    new_path_experiment = path_experiment + str(number) + "/"
    # ========== Elaborate and visualize the predicted images ====================
    test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
    full_img_height = test_imgs_orig.shape[2]
    full_img_width = test_imgs_orig.shape[3]
    if average_mode == True:
        pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
        orig_imgs = my_PreProc(test_imgs_orig[counter:counter + 1, :, :, :])  # originals
        gtruth_masks = test_groundtruth  # ground truth masks
    else:
        pred_imgs = recompone(pred_patches, 13, 12)  # predictions
        orig_imgs = recompone(patches_imgs_test, 13, 12)  # originals
        gtruth_masks = recompone(test_mask, 13, 12)  # masks


    orig_imgs = orig_imgs[:, :, 0:full_img_height, 0:full_img_width]
    pred_imgs = pred_imgs[:, :, 0:full_img_height, 0:full_img_width]
    pred_img = pred_imgs
    print(np.shape(pred_imgs),np.shape(test_mask),np.max(test_mask),np.min(test_mask))
    pred_imgs = np.array(list(map(lambda x: x[0] * x[1], zip(pred_imgs, test_mask))))

    threshold = float(config.get("testing settings", "vi_threshold"))
    if threshold:
        print("\nvisualized:  Costum threshold (for positive) of " + str(threshold))
        for i in range(pred_imgs.shape[2]):
            for j in range(pred_imgs.shape[3]):
                if pred_imgs[0][0][i][j] >= threshold:
                    pred_imgs[0][0][i][j] = 1
                else:
                    pred_imgs[0][0][i][j] = 0

    gtruth_masks = gtruth_masks[:, :, 0:full_img_height, 0:full_img_width]

    assert (orig_imgs.shape[0] == pred_imgs.shape[0] and orig_imgs.shape[0] == gtruth_masks.shape[0])
    N_predicted = orig_imgs.shape[0]
    group = N_visual
    assert (N_predicted % group == 0)

    for i in range(int(N_predicted / group)):
        orig_stripe = group_images(orig_imgs[i * group:(i * group) + group, :, :, :], group)
        masks_stripe = group_images(gtruth_masks[i * group:(i * group) + group, :, :, :], group)
        pred_stripe = group_images(pred_imgs[i * group:(i * group) + group, :, :, :], group)
        total_img = np.concatenate((orig_stripe, masks_stripe, pred_stripe), axis=0)
        visualize(total_img,
                  new_path_experiment + name_experiment + "_Original_GroundTruth_Prediction" + str(counter))  # .show()
    return pred_img, gtruth_masks


def evaluate_roc(y_true, y_scores, number, i):
    # Area under the ROC curve
    fpr, tpr, thresholds = roc_curve((y_true), y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
    print("\nArea under the ROC curve: " + str(AUC_ROC))
    # roc_curve = plt.figure()
    plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(path_experiment + "%d/ROC%d.png" % (number, i))

    return AUC_ROC


def evaluate_prc(y_true, y_scores, number, i):
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
    plt.savefig(path_experiment + "%d/Precision_recall%d.png" % (number, i))

    return AUC_prec_rec


def confusion_matrixd(y_true, y_scores):
    # Confusion matrix
    y_pred = np.empty((y_scores.shape[0]))
    threshold_confusion = 0.5
    print("\nConfusion matrix:  Costum threshold (for positive) of " + str(threshold_confusion))
    for i in range(y_scores.shape[0]):
        if y_scores[i] >= threshold_confusion:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
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


def evaluate(y_true, y_scores, number=0, i=0, result=[]):
    AUC_ROC = evaluate_roc(y_true, y_scores, number, i)
    AUC_prec_rec = evaluate_prc(y_true, y_scores, number, i)
    confusion, y_pred, accuracy, specificity, sensitivity, precision = confusion_matrixd(y_true, y_scores)
    jaccard_index = jacs(y_true, y_pred)
    F1_score = F1_s(y_true, y_pred)

    # Save the results
    file_perf = open(path_experiment + '/%d/performances%d.txt' % (number, i), 'w')
    file_perf.write("Area under the ROC curve: " + str(AUC_ROC)
                    + "\nArea under Precision-Recall curve: " + str(AUC_prec_rec)
                    + "\nJaccard similarity score: " + str(jaccard_index)
                    + "\nF1 score (F-measure): " + str(F1_score)
                    + "\n\nConfusion matrix:"
                    + str(confusion)
                    + "\nACCURACY: " + str(accuracy)
                    + "\nSENSITIVITY: " + str(sensitivity)
                    + "\nSPECIFICITY: " + str(specificity)
                    + "\nPRECISION: " + str(precision)
                    )
    file_perf.close()

    result[number][0] = result[number][0] + AUC_ROC
    result[number][1] = result[number][1] + AUC_prec_rec
    result[number][2] = result[number][2] + jaccard_index
    result[number][3] = result[number][3] + F1_score

    file_perf = open(path_experiment + '/%d/performances_all.txt' % number, 'w')
    file_perf.write("Area under the ROC curve: " + str(result[number][0])
                    + "\nArea under Precision-Recall curve: " + str(result[number][1])
                    + "\nJaccard similarity score: " + str(result[number][2])
                    + "\nF1 score (F-measure): " + str(result[number][3])
                    )
    file_perf.close()

    return result


def main(round, result):
    patches_imgs_test = None
    new_height = None
    new_width = None
    masks_test = None
    patches_masks_test = None
    predicision = None
    masks_test = None
    y_trues = None
    y_scores = None
    y_score = None
    y_true = None

    for i in range(round * full_images_to_test, (round + 1) * full_images_to_test):
        print("\n===============the %d/%d round===============" % (i, (round + 1) * full_images_to_test))
        if average_mode == True:
            patches_imgs_test, new_height, new_width, test_groundtruth, test_mask = get_data_testing_overlap(
                DRIVE_test_imgs_original=DRIVE_test_imgs_original,  # original
                DRIVE_test_groudTruth=path_data + config.get('data paths', 'test_groundTruth').replace("DRIVE",
                                                                                                       dataset),
                # masks
                DRIVE_test_mask=path_data + config.get('data paths', 'test_border_masks').replace("DRIVE",
                                                                                                       dataset),
                patch_height=patch_height,
                patch_width=patch_width,
                stride_height=stride_height,
                stride_width=stride_width,
                color_channel=int(config.get('public', 'color_channel')),
                number=i
            )
        else:
            patches_imgs_test, patches_masks_test = get_data_testing(
                DRIVE_test_imgs_original=DRIVE_test_imgs_original,  # original
                DRIVE_test_groudTruth=path_data + config.get('data paths', 'test_groundTruth').replace("DRIVE",
                                                                                                       dataset),
                # masks
                Imgs_to_test=int(config.get('testing settings', 'full_images_to_test')),
                patch_height=patch_height,
                patch_width=patch_width,
            )
        predicision = get_pred_patches(patches_imgs_test)

        for j in range(num_of_loss):
            number = j
            if not os.path.exists(path_experiment + str(number)):
                os.mkdir(path_experiment + str(number))
            if num_of_loss == 1:
                pred_patches = predicision
            else:
                pred_patches = predicision[j]

            pred_imgs, gtruth_masks = visualized(patches_imgs_test, test_mask, pred_patches, test_groundtruth,
                                                 new_height, new_width, number, i)

            score, true = pred_only_FOV(pred_imgs, gtruth_masks)  # returns data only inside the FOV

            if j == 0:
                y_score = score
                y_true = true
            else:
                y_score = np.concatenate((y_score, score), axis=1)
                y_true = np.concatenate((y_true, true), axis=1)

        if i % full_images_to_test == 0:
            y_scores = y_score
            y_trues = y_true
        else:
            y_scores = np.concatenate((y_scores, y_score), axis=0)
            y_trues = np.concatenate((y_trues, y_true), axis=0)

    for i in range(num_of_loss):
        number = i
        result = evaluate(y_trues[:, i:i + 1], y_scores[:, i:i + 1], number, round, result)

    return result


if __name__ == '__main__':
    result = []
    for i in range(num_of_loss):
        result.append([0, 0, 0, 0])
    for i in range(part):
        result = main(i, result)
