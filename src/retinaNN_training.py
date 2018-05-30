# coding:utf-8
import configparser
from keras.callbacks import ModelCheckpoint, Callback
import network
from extract_patches import get_data_training
from help_functions import *
import time
import os
import time
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# ========= Load settings from Config file
config = configparser.RawConfigParser()
config.read('./configuration.txt', encoding='utf-8')

# choose GPU
GPU = str(config.get('public', 'GPU'))
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
print("Using GPU ", GPU)

dataset = config.get('public', 'dataset')
path_data = config.get('data paths', 'path_local').replace("DRIVE", dataset)
name_experiment = config.get('experiment name', 'name')
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
net_name = config.get('public', 'net_config')

get_net = str(config.get('public', 'network'))
path_experiment = './' + name_experiment + "/"

configs = tf.ConfigProto()
configs.gpu_options.allow_growth = True
set_session(tf.Session(config=configs))


# save training information
def save_config():
    color_channel = "color_channel"
    if int(config.get('public', 'color_channel')) == 1:
        color_channel = "grey_scale map"
    elif int(config.get('public', 'color_channel')) == 3:
        color_channel = "color map"
    loss = "[" + config.get('public', 'loss_weight_0') + "," \
           + config.get('public', 'loss_weight_1') + "," \
           + config.get('public', 'loss_weight_2') + "," \
           + config.get('public', 'loss_weight_3') + "]"

    file = open(path_experiment + 'save_config.txt', 'w')
    file.write("Time of the experiment: " + str(time.ctime())
               + "\nUsing the network:" + config.get('public', 'network')
               + "\nUsing the dataset:" + dataset
               + "\ncolor_channel is:" + color_channel
               + "\nloss_weight is: " + loss)
    file.close()


def save_sample(patches_imgs_train, patches_masks_train):
    # ========= Save a sample of what you're feeding to the neural network ==========
    N_sample = min(patches_imgs_train.shape[0], 40)
    visualize(group_images(patches_imgs_train[0:N_sample, :, :, :], 5),
              './' + name_experiment + '/' + "sample_input_imgs")  # .show()
    visualize(group_images(patches_masks_train[0:N_sample, :, :, :], 5),
              './' + name_experiment + '/' + "sample_input_masks")  # .show()


# load the training network
def get_model(shape):
    n_ch = shape[1]
    patch_height = shape[2]
    patch_width = shape[3]

    if get_net == "unet":
        model = network.get_unet(n_ch, patch_height, patch_width)
    elif get_net == "hed":
        model = network.get_hed(n_ch, patch_height, patch_width)
    elif get_net == "unet3":
        model = network.get_unet3(n_ch, patch_height, patch_width)
    elif get_net == "unet4":
        model = network.get_unet4(n_ch, patch_height, patch_width)
    elif get_net == "unet5":
        model = network.get_unet5(n_ch, patch_height, patch_width)
    elif get_net == "unet_all":
        model = network.get_unet_all(n_ch, patch_height, patch_width)
    elif get_net == "unet_dm":
        model = network.get_unet_dm(n_ch, patch_height, patch_width)
    elif get_net == "unet_dm2":
        model = network.get_unet_dm2(n_ch, patch_height, patch_width)
    elif get_net == "unet_dsm":
        model = network.get_unet_dsm(n_ch, patch_height, patch_width)
    elif get_net == "unet_br":
        model = network.get_unet_br(n_ch, patch_height, patch_width)
    elif get_net == "unet_br2":
        model = network.get_unet_br2(n_ch, patch_height, patch_width)
    elif get_net == "fcnet":
        model = network.get_fcnet(n_ch, patch_height, patch_width)
    elif get_net == "unet_5l":
        model = network.get_unet5l(n_ch, patch_height, patch_width)
    elif get_net == "unet_6l":
        model = network.get_unet6l(n_ch, patch_height, patch_width)
    elif get_net == "unet_brnew":
        model = network.get_unet_brnew(n_ch, patch_height, patch_width)
    else:
        print("Please input a correct network!")
        exit(0)

    print("Check: final output of the network:", model.output_shape)
    json_string = model.to_json()
    open('./' + name_experiment + '/' + name_experiment + '_architecture.json', 'w').write(json_string)

    return model


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig('./' + name_experiment + '/' + name_experiment + '_loss.png')


def train():
    print("=====save config for training=====")
    save_config()

    print("\n=====get the training patches=====")
    patches_imgs_train, patches_masks_train = get_data_training(
        DRIVE_train_imgs_original=path_data + config.get('data paths', 'train_imgs_original').replace("DRIVE", dataset),
        DRIVE_train_groudTruth=path_data + config.get('data paths', 'train_groundTruth').replace("DRIVE", dataset),
        patch_height=int(config.get('data attributes', 'patch_height')),
        patch_width=int(config.get('data attributes', 'patch_width')),
        color_channel=int(config.get('public', 'color_channel')),
        train_coordinate=path_data + "/" + config.get('data paths', 'train_coordinate').replace("DRIVE", dataset),
        training_format=int(config.get(net_name, 'training_format')),
        num_of_loss=int(config.get(net_name, 'num_of_loss')),
        softmax_index=int(config.get(net_name, 'softmax_index')),
        differ_output=int(config.get(net_name, 'differ_output')))
    # save_sample(patches_imgs_train,patches_masks_train)

    print("")
    model = get_model(patches_imgs_train.shape)
    # ============  Training ==================================
    print("\n=====start training=====")
    stop_loss = 'val_loss'
    checkpointer = ModelCheckpoint(filepath='./' + name_experiment + '/' + name_experiment + '_best_weights.h5',
                                   verbose=1,
                                   monitor=stop_loss,
                                   mode='auto',
                                   save_best_only=True)  # save at each epoch if the validation decreased
    history = LossHistory()
    model.fit(patches_imgs_train, patches_masks_train,
              epochs=N_epochs, batch_size=batch_size, verbose=2,
              shuffle=True, validation_split=0.1, callbacks=[checkpointer, history])

    # ========== Save the last model ===================
    model.save_weights('./' + name_experiment + '/' + name_experiment + '_last_weights.h5', overwrite=True)
    history.loss_plot('epoch')


if __name__ == '__main__':
    start_time = time.time()
    train()
    print("\nTraining ended,it costs", str(int(time.time() - start_time)) + " seconds")
