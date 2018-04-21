import configparser
#Keras
import os
from keras.models import model_from_json
import sys

from PIL import Image as image
from pre_processing import *
from extract_patches import *
sys.path.insert(0, './lib/')

json_url = "./shenqi/shenqi_architecture.json"
weight_url = "./shenqi/shenqi_best_weights.h5"
image_url = "./shenqi/12.jpg"
patch_height = 48
patch_width = 48
stride_height = 5
stride_width = 5

num_of_loss = 3
masks_original = 1
softmax = 1


#========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('./configuration.txt')
name_experiment = config.get('experiment name', 'name')
path_experiment = './' +name_experiment +'/'


def get_pred_patches(patches_imgs_test):
    model = model_from_json(open(json_url).read())
    model.load_weights(weight_url)
    # Calculate the predictions
    print("patches_imgs_test.shape", patches_imgs_test.shape)
    pred_patches = model.predict(patches_imgs_test, batch_size=4, verbose=2)
    pred_patches = np.array(pred_patches)

    if masks_original == 0:
        pred = pred_to_imgs(pred_patches, patch_height, patch_width, "original")
    elif softmax:
        pred = []
        if num_of_loss == 1:
            pred = down_to_imgs(pred_patches,"original")
        else:
            for i in range(num_of_loss):
                pred_patch = down_to_imgs(pred_patches[i], "original")
                pred.append(pred_patch)
    else:
        pred = pred_patches

    pred_patches = np.array(pred)

    return pred_patches

def visualized(pred_patches, new_height, new_width, old_height,old_width,number):
    new_path_experiment = path_experiment + str(number) + "/"

    pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions

    pred_imgs = pred_imgs[:, :, 0:old_height, 0:old_width]

    pred_stripe = group_images(pred_imgs[0:1, :, :, :], 1)
    print("pred_stripe:", pred_stripe.shape)
    visualize(pred_stripe, new_path_experiment + name_experiment + "_Original_GroundTruth_Prediction"+str(if_reshape))

def get_data_testing(imgae_urls, patch_height,patch_width, stride_height, stride_width):
    test_img = image.open(imgae_urls)
    if if_reshape:
        test_img = test_img.resize((584,565))
    test_img = np.array(test_img)
    old_height = test_img.shape[0]
    old_width = test_img.shape[1]
    test = np.zeros(shape=(1,test_img.shape[2], test_img.shape[0], test_img.shape[1]))
    test_img = np.transpose(test_img, (2,0,1))
    test[0] = test_img
    test_img = my_PreProc(test)

    test_imgs = paint_border_overlap(test_img, patch_height, patch_width, stride_height, stride_width)
    patches_imgs_test = extract_ordered_overlap(test_imgs,patch_height,patch_width,stride_height,stride_width)

    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3],old_height,old_width

def main():
    patches_imgs_test, new_height, new_width, old_height,old_width= get_data_testing(
        imgae_urls= image_url,  # original
        patch_height=patch_height,
        patch_width=patch_width,
        stride_height=stride_height,
        stride_width=stride_width)

    predicision = get_pred_patches(patches_imgs_test)

    for j in range(num_of_loss):
        number = j
        if not os.path.exists(path_experiment + str(number)):
            os.mkdir(path_experiment + str(number))
        if num_of_loss == 1:
            pred_patches = predicision
        else:
            pred_patches = predicision[j]
        print(pred_patches.shape)

        visualized(pred_patches, new_height, new_width, old_height,old_width,number,)

if_reshape = 0
main()
print("======next======")
if_reshape = 1
main()


