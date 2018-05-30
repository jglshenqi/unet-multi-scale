# -*- coding: utf-8 -*-
import configparser
# Keras
import os
import tensorflow as tf
from keras.models import model_from_json
# from keras import backend as K
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
import time

# help_functions.py
from help_functions import *
from extract_patches import *
# pre_processing.py
from pre_processing import my_PreProc
import keras.backend.tensorflow_backend as KTF
from keras.backend.tensorflow_backend import set_session

plt.switch_backend('agg')
sys.path.insert(0, './lib/')

# ========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('./configuration.txt', encoding='utf-8')
# ===========================================
GPU = str(config.get('public', 'GPU'))
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
print("Using GPU ", GPU)
# run the training on invariant or local
dataset = config.get('public', 'dataset')
path_data = config.get('data paths', 'path_local').replace("DRIVE", dataset)
# original test images (for FOV selection)

full_images_to_test = int(config.get(dataset, 'full_images_to_test'))
# model name
name_experiment = config.get('experiment name', 'name')
path_experiment = './' + name_experiment + '/'
# ====== average mode ===========
net_name = config.get('public', 'net_config')
average_mode = config.getboolean('testing settings', 'average_mode')
num_of_loss = int(config.get(net_name, 'num_of_loss'))
softmax_index = int(config.get(net_name, 'softmax_index'))
part = int(config.get(dataset, "part"))
full_images_to_test = int(full_images_to_test / part)
differ_output = int(config.get(net_name, 'differ_output'))
get_net = str(config.get('public', 'network'))

configs = tf.ConfigProto()
configs.gpu_options.allow_growth = True
# configs.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=configs))

# K.clear_session()

if differ_output == 1:
    num_of_loss = 1


def get_pred_patches(patches_imgs_test):
    # ================ Run the prediction of the patches ==================================
    best_last = config.get('testing settings', 'best_last')

    model = model_from_json(open(path_experiment + name_experiment + '_architecture.json').read())
    model.load_weights(path_experiment + name_experiment + '_' + best_last + '_weights.h5')
    # Calculate the predictions
    print("patches_imgs_test.shape", patches_imgs_test.shape)
    pred_patches = model.predict(patches_imgs_test, batch_size=4, verbose=2)

    if softmax_index == 1:
        pred = []
        if differ_output == 1:
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


def visualized(pred_patches, test_mask, new_height, new_width, stride_height, stride_width, number, counter):
    new_path_experiment = path_experiment + str(number) + "/"
    # ========== Elaborate and visualize the predicted images ====================
    full_img_height = np.shape(test_mask)[2]
    full_img_width = np.shape(test_mask)[3]
    # turn pred patches into images
    pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)
    print("pred imgs shape:", pred_imgs.shape)

    pred_imgs = pred_imgs[:, :, 0:full_img_height, 0:full_img_width]
    # if it os necessary for showing the image in a threshold
    vi_threshold = float(config.get("testing settings", "vi_threshold"))
    if vi_threshold:
        print("\nvisualized:  Costum threshold (for positive) of " + str(vi_threshold))
        for i in range(pred_imgs.shape[2]):
            for j in range(pred_imgs.shape[3]):
                if pred_imgs[0][0][i][j] >= vi_threshold:
                    pred_imgs[0][0][i][j] = 1
                else:
                    pred_imgs[0][0][i][j] = 0
    pred_imgs = pred_imgs[0]
    pred_imgs = np.transpose(pred_imgs, (1, 2, 0))
    # save the pred images
    visualize(pred_imgs, new_path_experiment + name_experiment + "_Prediction_" + str(counter))


def predict():
    for i in range(0, full_images_to_test):
        print("\n===============the %d/%d round===============" % (i, full_images_to_test))
        # get the testing patches
        patches_imgs_test, test_mask, new_height, new_width = get_data_testing_overlap(
            DRIVE_test_imgs_original=path_data + config.get('data paths', 'test_imgs_original').replace("DRIVE",
                                                                                                        dataset),
            DRIVE_test_mask=path_data + config.get('data paths', 'test_border_masks').replace("DRIVE", dataset),
            start_img=i,
            patch_height=int(config.get('data attributes', 'patch_height')),
            patch_width=int(config.get('data attributes', 'patch_width')),
            stride_height=int(config.get('testing settings', 'stride_height')),
            stride_width=int(config.get('testing settings', 'stride_width')),
            color_channel=int(config.get('public', 'color_channel')),
            training_format=int(config.get(net_name, 'training_format')))
        # predict rhe testing patches
        predicision = get_pred_patches(patches_imgs_test)

        # turn the predictions into images
        for j in range(num_of_loss):
            number = j
            if not os.path.exists(path_experiment + str(number)):
                os.mkdir(path_experiment + str(number))
            if num_of_loss == 1:
                pred_patches = predicision
            else:
                pred_patches = predicision[j]

            visualized(pred_patches=pred_patches,
                       test_mask=test_mask,
                       new_height=new_height,
                       new_width=new_width,
                       stride_height=int(config.get('testing settings', 'stride_height')),
                       stride_width=int(config.get('testing settings', 'stride_width')),
                       number=number,
                       counter=i)


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

        for i in range(len(files)):
            file = name_experiment + "_Prediction_" + str(i) + ".png"
            # file = files[i]
            print(imgs_dir + file)
            img = Image.open(imgs_dir + file)
            img = np.array(img)

            if np.shape(img.shape)[0] == 2:
                img = img.reshape((img.shape[0], img.shape[1], 1))
            imgs[i] = img

        imgs = np.transpose(imgs, (0, 3, 1, 2))
        # show_img(imgs)

    return imgs


def prepare_data(urls_original, urls_save):
    for i in range(urls_original.__len__()):
        img = get_datasets(urls_original[i])
        write_hdf5(img, urls_save[i])


def evaluate_roc(y_true, y_scores, save_url):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # print(y_scores.max(),y_scores.min(),y_scores.sum())
    # print(y_true.max(), y_true.min(), y_true.sum())
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


def evaluate(y_true, y_scores, save_url):
    AUC_ROC = evaluate_roc(y_true, y_scores, save_url)
    AUC_prec_rec = evaluate_prc(y_true, y_scores, save_url)
    confusion, y_pred, accuracy, specificity, sensitivity, precision = confusion_matrixd(y_true, y_scores)
    jaccard_index = jacs(y_true, y_pred)
    F1_score = F1_s(y_true, y_pred)

    # Save the results
    file_perf = open(save_url + 'performance.txt', 'w')
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


def show_img(img):
    imgs = []
    for i in range(img.shape[0]):
        temp = img[i][0]
        imgs.extend(temp)
    print(np.shape(imgs))
    Image.fromarray(np.array(imgs)).show()


def evalu_roc(dataset, net, save_url):
    mask_url = path_data + config.get('data paths', 'test_border_masks').replace("DRIVE", dataset)

    gtruth_url = path_data + config.get('data paths', 'test_groundTruth').replace("DRIVE", dataset)
    pred_url = save_url + dataset + "_" + net + ".hdf5"

    y_true = load_hdf5(gtruth_url) / 255
    y_pred = load_hdf5(pred_url) / 255
    mask = load_hdf5(mask_url) / 255
    print("test", y_true.max(), y_pred.max(), mask.max(), y_true.min(), y_pred.min(), mask.min())
    print(y_true.shape, y_pred.shape, mask.shape)

    # show_img(y_true * 255)
    # show_img(y_pred * 255)
    # stop = input()

    score, true = pred_only_FOV(y_pred, y_true, mask)
    print(true.max(), true.min())

    evaluate(true, score, save_url)


if __name__ == '__main__':
    start_time = time.time()
    # get the pred images and save
    print("=======predict the images=======")
    # predict()
    print("Prediction ended,it costs", str(int(time.time() - start_time)) + " seconds")
    start_time = time.time()

    # turn the pred images into hdf5 files,just for last output
    print("=======transfer images to hdf5=======")
    url_original = [path_experiment + str(num_of_loss - 1) + "/"]
    url_save = [path_experiment + dataset + "_" + get_net + ".hdf5"]
    prepare_data(url_original, url_save)
    print("Transfering ended,it costs", str(int(time.time() - start_time)) + " seconds")
    start_time = time.time()

    # evaluate the results from hdf5 files
    print("=======evaluate=======")
    evalu_roc(dataset, get_net, path_experiment)
    print("Evaluation ended,it costs", str(int(time.time() - start_time)) + " seconds")
    start_time = time.time()
