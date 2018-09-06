import random
from help_functions import *
import skimage.measure

from pre_processing import my_PreProc, color_PreProc, show_image


# preprocessing for both training and testing
def preprocessing(data, color_channel):
    if color_channel == 1:
        data = my_PreProc(data)
    else:
        data = color_PreProc(data)
    data = data / 255
    return data


# adjust the training_gtruth if it is a deeply supervised problem
def adjust_gtruth(data, softmax_index, num_of_loss, differ_output):
    if softmax_index:
        data = np.concatenate((1 - data, data), axis=1)

    if differ_output:
        temp = [1, 2, 3, 4]
        temp[0] = skimage.measure.block_reduce(data, (1, 1, 4, 4), np.max)
        temp[1] = skimage.measure.block_reduce(data, (1, 1, 2, 2), np.max)
        temp[2] = data
        temp[3] = data
        print("train gtruth shape:", np.shape(temp[0]), np.shape(temp[1]), np.shape(temp[2]), np.shape(temp[3]),
              "train gtruth range: " + str(np.min(temp[3])) + ' - ' + str(np.max(temp[3])))
    else:
        temp = []
        for i in range(num_of_loss):
            temp.append(data)
        print("train gtruth shape:", np.shape(temp),
              "train gtruth range: " + str(np.min(temp)) + ' - ' + str(np.max(temp)))

    return temp


# Load the original data and return the extracted patches for training/testing
def get_data_training(DRIVE_train_imgs_original,
                      DRIVE_train_groudTruth,
                      patch_height,
                      patch_width,
                      color_channel,
                      train_coordinate,
                      training_format,
                      num_of_loss,
                      softmax_index,
                      differ_output):
    # load training images and groundtruth
    train_imgs = load_hdf5(DRIVE_train_imgs_original)
    train_gtruth = load_hdf5(DRIVE_train_groudTruth)

    # preprocessing of the images,result is between [0,1]
    train_imgs = preprocessing(train_imgs, color_channel)
    train_gtruth = train_gtruth / 255
    # show_image(train_imgs[0][0])

    print("train images shape:", train_imgs.shape,
          "train images range (min-max): " + str(np.min(train_imgs)) + ' - ' + str(np.max(train_imgs)))
    print("train gtruth shape:", train_gtruth.shape,
          "train gtruth range (min-max): " + str(np.min(train_gtruth)) + ' - ' + str(np.max(train_gtruth)))
    # stop = input()

    # extract the TRAINING patches from the full images
    patches_imgs_train = []
    patches_gtruth_train = []
    print(training_format)
    if training_format == 0:
        # patch_width = 49
        # patch_height = 49
        N_subimgs = 400000
        patches_imgs_train, patches_gtruth_train = extract_random(train_imgs,
                                                                  train_gtruth,
                                                                  patch_height,
                                                                  patch_width,
                                                                  N_subimgs)
    elif training_format == 1:
        patches_imgs_train, patches_gtruth_train = extract_coordinate(train_imgs,
                                                                      train_gtruth,
                                                                      patch_height,
                                                                      patch_width,
                                                                      train_coordinate)
    elif training_format == 2:
        print(train_imgs.shape)

        patches_imgs_train = train_imgs
        patches_gtruth_train = train_gtruth

    print("train patches shape:", np.shape(patches_imgs_train),
          "train patches range: " + str(np.min(patches_imgs_train)) + ' - ' + str(np.max(patches_imgs_train)))
    patches_gtruth_train = adjust_gtruth(patches_gtruth_train, softmax_index, num_of_loss, differ_output)

    return patches_imgs_train, patches_gtruth_train


def get_data_testing_overlap(DRIVE_test_imgs_original,
                             DRIVE_test_mask,
                             start_img,
                             patch_height,
                             patch_width,
                             stride_height,
                             stride_width,
                             color_channel,
                             training_format):
    # load images from hdf5 file
    test_imgs_original = load_hdf5(DRIVE_test_imgs_original)
    print("test imgs_original shape:", test_imgs_original.shape)
    test_mask = load_hdf5(DRIVE_test_mask)
    # preprocessing of the test
    test_imgs = preprocessing(test_imgs_original, color_channel)
    test_mask = test_mask / 255

    test_imgs = test_imgs[start_img:start_img + 1, :, :, :]
    test_mask = test_mask[start_img:start_img + 1, :, :, :]

    print("test imgs shape:", test_imgs.shape, "test imgs range:", np.min(test_imgs), "-", np.max(test_imgs))
    print("test mask shape:", test_mask.shape, "test mask range:", np.min(test_mask), "-", np.max(test_mask))
    # get the patches for test in different conditions
    patches_imgs_test = []
    if training_format == 0:
        # patch_width = 49
        # patch_height = 49
        stride_height = 1
        stride_width = 1
        patches_imgs_test = extract_ordered_overlap(test_imgs,
                                                    patch_height,
                                                    patch_width,
                                                    stride_height,
                                                    stride_width)
    elif training_format == 1:
        test_imgs = paint_border_overlap(test_imgs,
                                         patch_height,
                                         patch_width,
                                         stride_height,
                                         stride_width)
        patches_imgs_test = extract_ordered_overlap(test_imgs,
                                                    patch_height,
                                                    patch_width,
                                                    stride_height,
                                                    stride_width)
    elif training_format == 2:
        patches_imgs_test = test_imgs

    # print("test imgs shape:", test_imgs.shape, "test imgs range:", np.min(test_imgs), "-", np.max(test_imgs))
    print("patches_imgs_test shape:", patches_imgs_test.shape,
          "patches_imgs_test range:", np.min(patches_imgs_test), "-", np.max(patches_imgs_test))

    return patches_imgs_test, test_mask, test_imgs.shape[2], test_imgs.shape[3]


# get the training patches,just for the fcnet
def extract_random(full_imgs, full_gtruth, patch_h, patch_w, N_patches):
    patches = np.empty((N_patches, full_imgs.shape[1], patch_h, patch_w))
    patches_gtruth = np.empty((N_patches, 1))
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    patch_per_img = int(N_patches / full_imgs.shape[0])  # N_patches equally divided in the full images
    iter_tot = 0  # iter over the total numbe rof patches (N_patches)

    for i in range(full_imgs.shape[0]):  # loop over the full images
        k = 0
        while k < patch_per_img:
            # delt = int(patch_w / 2)
            x_center = random.randint(0 + int((patch_w) / 2), img_w - int((patch_w) / 2) - 1)
            y_center = random.randint(0 + int((patch_h) / 2), img_h - int((patch_h) / 2) - 1)
            patch = full_imgs[i, :, y_center - int(patch_h / 2):y_center + int(patch_h / 2) + 1,
                    x_center - int(patch_w / 2):x_center + int(patch_w / 2) + 1]
            patch_gtruth = full_gtruth[i, :, y_center, x_center]
            patches[iter_tot] = patch
            patches_gtruth[iter_tot] = patch_gtruth
            iter_tot += 1  # total
            k += 1  # per full_img
    return patches, patches_gtruth


# get the training patches from coordinate file
def extract_coordinate(full_imgs, full_gtruth, patch_h, patch_w, train_coordinate):
    usage = 1
    training_node = load_node(train_coordinate)
    training_shape = np.shape(training_node)
    assert (len(full_imgs.shape) == 4 and len(full_gtruth.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3
    assert (full_gtruth.shape[1] == 1)  # masks only black and white
    assert (full_imgs.shape[2] == full_gtruth.shape[2] and full_imgs.shape[3] == full_gtruth.shape[3])
    patches = np.empty((int(training_shape[0] * training_shape[1] / usage), full_imgs.shape[1], patch_h, patch_w))
    patches_gtruth = np.empty(
        (int(training_shape[0] * training_shape[1] / usage), full_gtruth.shape[1], patch_h, patch_w))
    patch_per_img = int(
        training_shape[0] * training_shape[1] / full_imgs.shape[0])  # N_patches equally divided in the full images
    print("patches per full image: " + str(patch_per_img))
    iter_tot = 0  # iter over the total numbe rof patches (N_patches)

    for i in range(training_shape[0]):  # loop over the full images
        k = 0
        while k < training_shape[1]:
            if k % usage != 0:
                k = k + 1
                continue
            x_center = training_node[i][k][0]
            y_center = training_node[i][k][1]

            patch = full_imgs[i, :, y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                    x_center - int(patch_w / 2):x_center + int(patch_w / 2)]
            patch_gtruth = full_gtruth[i, :, y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                           x_center - int(patch_w / 2):x_center + int(patch_w / 2)]
            patches[iter_tot] = patch
            patches_gtruth[iter_tot] = patch_gtruth
            iter_tot += 1  # total
            k += 1  # per full_img

    return patches, patches_gtruth


# paint border if it is not exact division by stride
def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    leftover_h = (img_h - patch_h) % stride_h  # leftover on the h dim
    leftover_w = (img_w - patch_w) % stride_w  # leftover on the w dim
    if (leftover_h != 0):  # change dimension of img_h
        print("\nthe side H is not compatible with the selected stride of " + str(stride_h))
        print("img_h " + str(img_h) + ", patch_h " + str(patch_h) + ", stride_h " + str(stride_h))
        print("(img_h - patch_h) MOD stride_h: " + str(leftover_h))
        print("So the H dim will be padded with additional " + str(stride_h - leftover_h) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0], full_imgs.shape[1], img_h + (stride_h - leftover_h), img_w))
        tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:img_h, 0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    if (leftover_w != 0):  # change dimension of img_w
        print("the side W is not compatible with the selected stride of " + str(stride_w))
        print("img_w " + str(img_w) + ", patch_w " + str(patch_w) + ", stride_w " + str(stride_w))
        print("(img_w - patch_w) MOD stride_w: " + str(leftover_w))
        print("So the W dim will be padded with additional " + str(stride_w - leftover_w) + " pixels")
        tmp_full_imgs = np.zeros(
            (full_imgs.shape[0], full_imgs.shape[1], full_imgs.shape[2], img_w + (stride_w - leftover_w)))
        tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:full_imgs.shape[2], 0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    print("new full images shape: \n" + str(full_imgs.shape))
    return full_imgs


# Divide all the full_imgs in pacthes
def extract_ordered_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    assert ((img_h - patch_h) % stride_h == 0 and (img_w - patch_w) % stride_w == 0)
    N_patches_img = ((img_h - patch_h) // stride_h + 1) * (
        (img_w - patch_w) // stride_w + 1)  # // --> division between integers
    N_patches_tot = N_patches_img * full_imgs.shape[0]
    print("Number of patches on h : " + str(((img_h - patch_h) // stride_h + 1)))
    print("Number of patches on w : " + str(((img_w - patch_w) // stride_w + 1)))
    print("number of patches per image: " + str(N_patches_img) + ", totally for this dataset: " + str(N_patches_tot))
    patches = np.empty((N_patches_tot, full_imgs.shape[1], patch_h, patch_w))
    iter_tot = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                patch = full_imgs[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w]
                patches[iter_tot] = patch
                iter_tot += 1  # total
    assert (iter_tot == N_patches_tot)
    return patches  # array with all the full_imgs divided in patches


def recompone_image(pred_patches,full_img_height,full_img_width,patch_height,patch_width):
    temp = int(patch_height/2)
    k = 0
    img = np.zeros(shape=(1,1,full_img_height,full_img_width))
    for i in range(temp, full_img_height - temp):
        for j in range(temp, full_img_width-temp):
            img[0][0][i][j] = pred_patches[k]
            k = k+1
    print(k,np.sum(pred_patches))
    return img



# recover the predictions if it is extended
def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    assert (len(preds.shape) == 4)  # 4D arrays
    assert (preds.shape[1] == 1 or preds.shape[1] == 3)  # check the channel is 1 or 3
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h - patch_h) // stride_h + 1
    N_patches_w = (img_w - patch_w) // stride_w + 1
    N_patches_img = N_patches_h * N_patches_w
    print("N_patches_h: " + str(N_patches_h), end=" ")
    print("N_patches_w: " + str(N_patches_w), end=" ")
    print("N_patches_img: " + str(N_patches_img), end=" ")
    assert (preds.shape[0] % N_patches_img == 0)
    N_full_imgs = preds.shape[0] // N_patches_img
    print("According to the dimension inserted, there are " + str(N_full_imgs) + " full images (of " + str(
        img_h) + "x" + str(img_w) + " each)")
    full_prob = np.zeros(
        (N_full_imgs, preds.shape[1], img_h, img_w))  # itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))

    k = 0  # iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                full_prob[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w] += preds[
                    k]
                full_sum[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w] += 1
                k += 1
    assert (k == preds.shape[0])
    assert (np.min(full_sum) >= 1.0)  # at least one
    final_avg = full_prob / full_sum
    assert (np.max(final_avg) <= 1.0)  # max value for a pixel is 1.0
    assert (np.min(final_avg) >= 0.0)  # min value for a pixel is 0.0
    return final_avg


# Recompone the full images with the patches
def recompone(data, N_h, N_w):
    assert (data.shape[1] == 1 or data.shape[1] == 3)  # check the channel is 1 or 3
    assert (len(data.shape) == 4)
    N_pacth_per_img = N_w * N_h
    assert (data.shape[0] % N_pacth_per_img == 0)
    N_full_imgs = data.shape[0] / N_pacth_per_img
    patch_h = data.shape[2]
    patch_w = data.shape[3]
    N_pacth_per_img = N_w * N_h
    # define and start full recompone
    full_recomp = np.empty((N_full_imgs, data.shape[1], N_h * patch_h, N_w * patch_w))
    k = 0  # iter full img
    s = 0  # iter single patch
    while s < data.shape[0]:
        # recompone one:
        single_recon = np.empty((data.shape[1], N_h * patch_h, N_w * patch_w))
        for h in range(N_h):
            for w in range(N_w):
                single_recon[:, h * patch_h:(h * patch_h) + patch_h, w * patch_w:(w * patch_w) + patch_w] = data[s]
                s += 1
        full_recomp[k] = single_recon
        k += 1
    assert (k == N_full_imgs)
    return full_recomp


# Extend the full images becasue patch divison is not exact
def paint_border(data, patch_h, patch_w):
    assert (len(data.shape) == 4)  # 4D arrays
    assert (data.shape[1] == 1 or data.shape[1] == 3)  # check the channel is 1 or 3
    img_h = data.shape[2]
    img_w = data.shape[3]
    new_img_h = 0
    new_img_w = 0
    if (img_h % patch_h) == 0:
        new_img_h = img_h
    else:
        new_img_h = ((int(img_h) / int(patch_h)) + 1) * patch_h
    if (img_w % patch_w) == 0:
        new_img_w = img_w
    else:
        new_img_w = ((int(img_w) / int(patch_w)) + 1) * patch_w
    new_data = np.zeros((data.shape[0], data.shape[1], new_img_h, new_img_w))
    new_data[:, :, 0:img_h, 0:img_w] = data[:, :, :, :]
    return new_data


# return only the pixels contained in the FOV, for both images and masks
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
