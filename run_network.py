# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import configparser

sys.path.append('./src/network')

# get the config
config = configparser.RawConfigParser()
config.read('./configuration.txt', encoding='utf-8')

name_experiment = config.get('experiment name', 'name')
training = config.getboolean('pre configs', 'training')
train_nohup = config.getboolean('pre configs', 'train_nohup')
testing = config.getboolean('pre configs', 'testing')
test_nohup = config.getboolean('pre configs', 'test_nohup')
training_file = config.get('pre configs', 'training_file')
num_for_training = 1
if training:
    num_for_training = int(config.get('pre configs', 'num_for_training'))


# train the network
def train_network():
    if train_nohup:
        print("\n2. Run the training on GPU with nohup")
        os.system(
            'nohup python -u' + training_file + '> ' + './' + name_experiment + '/' + name_experiment + '_training.nohup')
    else:
        print("\n2. Run the training on GPU (no nohup)")
        print(training_file)
        os.system('python ' + training_file)


# test the network
def test_network():
    if test_nohup:
        print("\n2. Run the prediction on GPU  with nohup")
        os.system(
            'python -u ./src/retinaNN_predict.py > ' + './' + name_experiment + '/' + name_experiment + '_prediction.txt')
    else:
        print("\n2.Run the prediction on GPU (no nohup)")
        os.system('python ./src/retinaNN_predict.py')


def start_train():
    result_dir = name_experiment
    print("\n1. Create directory for the results (if not already existing)")
    if os.path.exists(result_dir):
        print("Dir already existing")
    elif sys.platform == 'win32':
        os.system('mkdir ' + result_dir)
    else:
        os.system('mkdir -p ' + result_dir)

    if training:
        train_network()
    if testing:
        test_network()

    if num_for_training > 1:
        os.rename("shenqi", str(i))


if __name__ == '__main__':
    # os.system('python ./src/prepare_data.py')
    for i in range(num_for_training):
        start_train()
