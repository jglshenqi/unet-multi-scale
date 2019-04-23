# coding:utf-8
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from extract_patches import adjust_gtruth
from help_functions import get_model, visualize, group_images, load_hdf5
from matplotlib import pyplot as plt
import sys
import time
import get_config
import numpy as np

sys.path.append("./src/network/")
config = get_config.get_config()


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
        self.loss_plot('epoch')

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
        plt.savefig('./' + config.name_experiment + '/' + config.name_experiment + '_loss.png')


class back_calling():
    def __init__(self, filepath, save_best):
        stop_loss = 'val_loss'
        self.checkpointer = ModelCheckpoint(filepath=filepath,
                                            verbose=1,
                                            monitor=stop_loss,
                                            mode='auto',
                                            save_best_only=save_best)  # save at each epoch if the validation decreased

        self.history = LossHistory()
        self.tensorboard = TensorBoard(log_dir="./shenqi",
                                       histogram_freq=1,
                                       write_graph=True)


def train():
    print("=====save config for training=====")
    config.save_config()
    config.choose_GPU()
    print("")

    print("\n=====get the training patches=====")
    train_patches = config.path_data + "train_patch" + str(config.patch_width) + "_" + str(config.N_sampling) + ".hdf5"
    groundtruth_patches = train_patches.replace("train_patch", "groundtruth_patch")
    patches_imgs_train = np.array(load_hdf5(train_patches))
    patches_groundtruth_train = np.array(load_hdf5(groundtruth_patches))
    print("The shape of training patches is:", patches_imgs_train.shape,
          "train images range: " + str(patches_imgs_train.min()) + ' - ' + str(patches_imgs_train.max()))
    patches_groundtruth_train = adjust_gtruth(patches_groundtruth_train, config.softmax_index, config.num_of_loss)
    print("The shape of groundtruth patches is:", np.shape(patches_groundtruth_train),
          "train gtruth range: " + str(np.min(patches_groundtruth_train)) + ' - '
          + str(np.max(patches_groundtruth_train)))
    print("")

    print("\n=====load the network=====")
    save_url = './' + config.name_experiment + '/' + config.name_experiment + '_architecture.json'
    model = get_model(patches_imgs_train.shape, config, save_url)
    print("")

    print("\n=====start training=====")
    # filepath = './' + config.name_experiment + '/' + config.name_experiment + '-{epoch:02d}-weights.h5'
    filepath = './' + config.name_experiment + '/' + config.name_experiment + '_best_weights.h5'
    back_call = back_calling(filepath, save_best=True)
    model.fit(patches_imgs_train,
              patches_groundtruth_train,
              epochs=config.N_epochs,
              batch_size=config.batch_size,
              verbose=2,
              shuffle=True,
              validation_split=0.1,
              callbacks=[back_call.history, back_call.checkpointer])
    # history.loss_plot('epoch')

    # ========== Save the last model ===================
    model.save_weights('./' + config.name_experiment + '/' + config.name_experiment + '_last_weights.h5',
                       overwrite=True)


if __name__ == '__main__':
    start_time = time.time()
    train()
    print("\nTraining ended,it costs", str(int(time.time() - start_time)) + " seconds")
