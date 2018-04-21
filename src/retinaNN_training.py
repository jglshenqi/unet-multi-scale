import configparser
from keras.callbacks import ModelCheckpoint, Callback
import network
from extract_patches import get_data_training
from help_functions import *
from scipy import ndimage
import skimage.measure

# ========= Load settings from Config file
config = configparser.RawConfigParser()
config.read('./configuration.txt')
dataset = config.get('public', 'dataset')
# patch to the datasets
path_data = config.get('data paths', 'path_local').replace("DRIVE", dataset)
# Experiment name
name_experiment = config.get('experiment name', 'name')
# training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
net_name = config.get('public', 'network')
num_of_classes = int(config.get('public', 'num_of_classes'))
num_of_loss = int(config.get(net_name, 'num_of_loss'))
softmax = int(config.get(net_name, 'softmax'))
mask_original = int(config.get(net_name, 'mask_original'))
type_of_output = int(config.get(net_name, 'type_of_output'))


def save_sample(patches_imgs_train, patches_masks_train):
    # ========= Save a sample of what you're feeding to the neural network ==========
    N_sample = min(patches_imgs_train.shape[0], 40)
    visualize(group_images(patches_imgs_train[0:N_sample, :, :, :], 5),
              './' + name_experiment + '/' + "sample_input_imgs")  # .show()
    visualize(group_images(patches_masks_train[0:N_sample, :, :, :], 5),
              './' + name_experiment + '/' + "sample_input_masks")  # .show()


def get_model(shape):
    # =========== Construct and save the model arcitecture =====
    n_ch = shape[1]
    print("for test", n_ch)
    patch_height = shape[2]
    patch_width = shape[3]

    # model = network.get_dianet(n_ch, patch_height, patch_width) #the U-net model
    model = network.get_unet_all(n_ch, patch_height, patch_width)
    print("Check: final output of the network:")
    print(model.output_shape)
    # plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
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


def main():
    patches_imgs_train, patches_masks_train = get_data_training(
        DRIVE_train_imgs_original=path_data + config.get('data paths', 'train_imgs_original').replace("DRIVE", dataset),
        DRIVE_train_groudTruth=path_data + config.get('data paths', 'train_groundTruth').replace("DRIVE", dataset),
        # masks
        patch_height=int(config.get('data attributes', 'patch_height')),
        patch_width=int(config.get('data attributes', 'patch_width')),
        N_subimgs=int(config.get('training settings', 'N_subimgs')),
        inside_FOV=config.getboolean('training settings', 'inside_FOV'),
        # select the patches only inside the FOV  (default == True))
        color_channel=int(config.get('public', 'color_channel')),
        train_coordinate=path_data + "/" + config.get('data paths', 'train_coordinate').replace("DRIVE", dataset))

    print(np.max(patches_masks_train), np.max(patches_imgs_train))

    # save_sample(patches_imgs_train,patches_masks_train)

    model = get_model(patches_imgs_train.shape)

    # ============  Training ==================================
    checkpointer = ModelCheckpoint(filepath='./' + name_experiment + '/' + name_experiment + '_best_weights.h5',
                                   verbose=1,
                                   monitor='val_loss', mode='auto',
                                   save_best_only=True)  # save at each epoch if the validation decreased

    if mask_original == 0:
        patches_masks_train = masks_Unet(patches_masks_train)  # reduce memory consumption
    else:
        if softmax:
            patches_masks_train = masks_appsoft(patches_masks_train)
        p = []
        if type_of_output == 0:
            for i in range(num_of_loss):
                p.append(patches_masks_train)
        elif type_of_output == 1:
            p = [1, 2, 3, 4]
            p[0] = skimage.measure.block_reduce(patches_masks_train, (1, 1, 4, 4), np.max)
            p[1] = skimage.measure.block_reduce(patches_masks_train, (1, 1, 2, 2), np.max)
            p[2] = patches_masks_train
            p[3] = patches_masks_train
        patches_masks_train = p

    history = LossHistory()

    model.fit(patches_imgs_train, patches_masks_train,
              # np.array(patches_masks_train[0]),
              epochs=N_epochs, batch_size=batch_size, verbose=2,
              shuffle=True, validation_split=0.1, callbacks=[checkpointer, history])

    # ========== Save and test the last model ===================
    model.save_weights('./' + name_experiment + '/' + name_experiment + '_last_weights.h5', overwrite=True)

    history.loss_plot('epoch')


if __name__ == '__main__':
    main()
