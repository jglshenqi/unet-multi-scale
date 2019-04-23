import configparser
import time
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = configparser.RawConfigParser()
config.read('./configuration.txt', encoding='utf-8')


# get all the config from configuration
class get_config():
    def __init__(self):
        # get public parameter
        self.network_file = str(config.get('public', 'network_file'))
        self.get_net = str(config.get('public', 'network'))
        self.net_config = config.get('public', 'net_config')
        self.GPU = str(config.get('public', 'GPU'))
        self.dataset = config.get('public', 'dataset')
        self.defalt_parameter = int(config.get('public', 'defalt_parameter'))

        # get the path of data
        path_data = config.get('data paths', 'path_local')
        self.path_data = path_data.replace("DRIVE", self.dataset)

        train_imgs_original = self.path_data + config.get('data paths', 'train_imgs_original')
        train_groudTruth = self.path_data + config.get('data paths', 'train_groundTruth')
        train_mask = self.path_data + config.get('data paths', 'train_border_masks')
        test_imgs_original = self.path_data + config.get('data paths', 'test_imgs_original')
        test_groundtruth = self.path_data + config.get('data paths', 'test_groundTruth')
        test_mask = self.path_data + config.get('data paths', 'test_border_masks')
        train_coordinate = config.get('data paths', 'train_coordinate')
        train_patches = self.path_data + config.get('data paths', 'train_patches')
        groundtruth_patches = self.path_data + config.get('data paths', 'groundtruth_patches')

        self.train_imgs_original = train_imgs_original.replace("DRIVE", self.dataset)
        self.train_groudTruth = train_groudTruth.replace("DRIVE", self.dataset)
        self.train_mask = train_mask.replace("DRIVE", self.dataset)
        self.test_imgs_original = test_imgs_original.replace("DRIVE", self.dataset)
        self.test_groundtruth = test_groundtruth.replace("DRIVE", self.dataset)
        self.test_mask = test_mask.replace("DRIVE", self.dataset)
        self.train_coordinate = train_coordinate.replace("DRIVE", self.dataset)
        self.train_patches = train_patches.replace("DRIVE", self.dataset)
        self.groundtruth_patches = groundtruth_patches.replace("DRIVE", self.dataset)

        # get the name of the experiment
        self.name_experiment = config.get('experiment name', 'name')

        # get the data parameter
        self.patch_height = int(config.get('data attributes', 'patch_height'))
        self.patch_width = int(config.get('data attributes', 'patch_width'))
        self.N_sampling = int(config.get('data attributes', 'N_sampling'))
        self.stride_height = int(config.get('data attributes', 'stride_height'))
        self.stride_width = int(config.get('data attributes', 'stride_width'))
        self.full_images_to_train = int(config.get(self.dataset, 'full_images_to_train'))
        self.full_images_to_test = int(config.get(self.dataset, 'full_images_to_test'))
        self.dataset_mean = float(config.get(self.dataset, 'dataset_mean'))
        self.dataset_std = float(config.get(self.dataset, 'dataset_std'))

        # get the training parameter
        self.N_epochs = int(config.get('training settings', 'N_epochs'))
        self.batch_size = int(config.get('training settings', 'batch_size'))

        self.training_format = int(config.get('training settings', 'training_format'))
        self.num_of_loss = int(config.get('training settings', 'num_of_loss'))
        self.softmax_index = int(config.get('training settings', 'softmax_index'))

        # get the test parameter
        self.best_last = config.get('testing settings', 'best_last')
        self.average_mode = config.getboolean('testing settings', 'average_mode')
        self.vi_threshold = float(config.get("testing settings", "vi_threshold"))

        if self.defalt_parameter:
            self.get_defalt_parameter()

    # save training information
    def save_config(self):
        path_experiment = './' + self.name_experiment + "/"
        file = open(path_experiment + 'save_config.txt', 'w')
        file.write("Time of the experiment: " + str(time.ctime())
                   + "\r\nUsing the network:" + self.network_file + "  " + self.get_net
                   + "\r\nUsing the dataset:" + self.dataset
                   + "\r\ntrain patch:" + str(self.patch_width)+"_"+str(self.N_sampling))
        file.close()

    def choose_GPU(self):
        # choose GPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = self.GPU
        print("Using GPU ", self.GPU)

        configs = tf.ConfigProto()
        configs.gpu_options.allow_growth = True
        set_session(tf.Session(config=configs))

    def get_defalt_parameter(self):
        self.training_format = int(config.get(self.net_config, 'training_format'))
        self.num_of_loss = int(config.get(self.net_config, 'num_of_loss'))
        self.softmax_index = int(config.get(self.net_config, 'softmax_index'))
