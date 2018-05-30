# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import configparser

config = configparser.RawConfigParser()
config.readfp(open(r'./configuration.txt', encoding='utf-8'))
name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('testing settings', 'nohup')  # std output on log file?

run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

# create a folder for the results if not existing already
result_dir = name_experiment
print("\n1. Create directory for the results (if not already existing)")
if os.path.exists(result_dir):
    pass
elif sys.platform == 'win32':
    os.system('md ' + result_dir)
else:
    os.system('mkdir -p ' + result_dir)

# finally run the prediction
if nohup:
    print("\n2. Run the prediction on GPU  with nohup")
    os.system(
        'python -u ./src/retinaNN_predict.py > ' + './' + name_experiment + '/' + name_experiment + '_prediction.txt')
else:
    print("\n2.Run the prediction on GPU (no nohup)")
    os.system('python ./src/retinaNN_predict.py')
