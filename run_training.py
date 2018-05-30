# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import configparser

config = configparser.RawConfigParser()
config.readfp(open(r'./configuration.txt', encoding='utf-8'))
name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('training settings', 'nohup')  # std output on log file?

# create a folder for the results
result_dir = name_experiment
print("\n1. Create directory for the results (if not already existing)")
if os.path.exists(result_dir):
    print("Dir already existing")
elif sys.platform == 'win32':
    os.system('mkdir ' + result_dir)
else:
    os.system('mkdir -p ' + result_dir)

print("copy the configuration file in the results folder")
if sys.platform == 'win32':
    os.system('copy configuration.txt .\\' + name_experiment + '\\' + name_experiment + '_configuration.txt')
    os.system('cp configuration.txt ./' + name_experiment + '/' + name_experiment + '_configuration.txt')

# run the experiment
if nohup:
    print("\n2. Run the training on GPU with nohup")
    os.system(
        'nohup python -u ./src/retinaNN_training.py > ' + './' + name_experiment + '/' + name_experiment + '_training.nohup')
else:
    print("\n2. Run the training on GPU (no nohup)")
    os.system('python ./src/retinaNN_training.py')
