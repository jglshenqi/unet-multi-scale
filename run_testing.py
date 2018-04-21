###################################################
#
#   Script to execute the prediction
#
##################################################

import os, sys
import configparser

config = configparser.RawConfigParser()
config.readfp(open(r'./configuration.txt'))

GPU = str(config.get('public', 'GPU'))
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

print("Using GPU ",GPU)

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

config = configparser.RawConfigParser()
config.readfp(open(r'./configuration.txt'))
name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('testing settings', 'nohup')   #std output on log file?

run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

#create a folder for the results if not existing already
result_dir = name_experiment
print ("\n1. Create directory for the results (if not already existing)")
if os.path.exists(result_dir):
    pass
elif sys.platform=='win32':
    os.system('md ' + result_dir)
else:
    os.system('mkdir -p ' + result_dir)

# finally run the prediction
if nohup:
    print ("\n2. Run the prediction on GPU  with nohup")
    os.system(run_GPU +' python -u ./src/retinaNN_predict.py > ' +'./'+name_experiment+'/'+name_experiment+'_prediction.txt')
else:
    print ("\n2. Run the prediction on GPU (no nohup)")
    os.system(run_GPU +' python ./src/retinaNN_predict.py')
