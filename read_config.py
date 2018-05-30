import configparser
from keras.callbacks import ModelCheckpoint, Callback
import network
from extract_patches import get_data_training
from help_functions import *
import skimage.measure
