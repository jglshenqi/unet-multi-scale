import numpy as np
import random
import h5py
import pickle
from PIL import Image
from skimage import io
import threading
from multiprocessing import Manager, Pool
import multiprocessing
import os, time, random
import time
import threading
from random import random


# load images from hdf5 files
def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["image"][()]


def load_node(url):
    with open(url, 'rb') as f:
        list_ = pickle.load(f)
    return list_


class load_data():
    def __init__(self):
        self.q = Manager().Queue(1000)
        print("stop")
        self.get_data()

    def get_data(self):
        N_sampling = 200
        path = "../dataset/DRIVE/DRIVE_datasets_training_testing/" + "train_patch_48_" + str(N_sampling)
        patch_names = load_node(path + "/patch_names.pickle")
        pool = Pool(processes=6)
        pool.apply_async(self.write(patch_names))
        pool.apply_async(self.read())

    def write(self, patch_names):
        while True:
            # print("i am writing", patch_names)
            np.random.shuffle(patch_names)
            for line in patch_names:
                # while self.q.full():
                #     time.sleep(1)
                if self.q.full():
                    continue
                self.q.put(line)
                print("i am writing", line)

    def read(self):
        while True:
            print("i am reading")
            while self.q.empty():
                time.sleep(1)
            self.q.get()
            # print(self.q.get())


def generate_train_arrays_from_queue():
    validation_split = 0.9

    N_sampling = 200

    path = "../dataset/DRIVE/DRIVE_datasets_training_testing/" + "train_patch_48_" + str(N_sampling)
    patch_names = load_node(path + "/patch_names.pickle")
    print(np.shape(patch_names))

    patch_names = patch_names[:int(len(patch_names) * validation_split)]
    print(np.shape(patch_names))
    q = Manager().Queue(100)
    p = Pool()
    p.apply_async(write, args=(q, path, patch_names,))
    temp = p.apply_async(read, args=(q,))
    print(temp.get())
    stop = input()
    # p.close()
    # p.join()
    print("stop")


def func(msg):
    return multiprocessing.current_process().name + '-' + msg


def double(n):
    return n * 2


class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None


def producer():
    while True:
        np.random.shuffle(patch_names)
        print(patch_names)
        for line in patch_names:
            xy = []
            while q.full():
                print("have a rest")
                time.sleep(1)
            x_url = path_url + line[0]
            y_url = path_url + line[1]
            xy.append(load_hdf5(x_url)[0])
            xy.append(load_hdf5(y_url)[0])
            q.put(xy)


def consumer():
    while True:
        # print("test")
        # yield q.get()
        x = y = X = Y = []
        for i in range(batch_size):
            while q.empty():
                print("have another rest")
                time.sleep(1)
            xy = q.get()
            x.append(xy[0:1])
            y.append(xy[1:2])
        y = np.array(y)
        y = np.concatenate((1 - y, y), axis=1)
        for i in range(num_of_loss):
            Y.append(np.array(y))
        yield X, Y




if __name__ == "__main__":
    path = "../dataset/DRIVE/DRIVE_datasets_training_testing/"
    batch_size = 32
    validation_split = 0.9
    num_of_loss = 4
    N_sampling = 200
    path_url = path + "train_patch_48_" + str(N_sampling)
    patch_names = load_node(path_url + "/patch_names.pickle")

    q = Manager().Queue(100)
    a = MyThread(producer,args=())
    a.start()
    b = MyThread(consumer,args=())
    b.start()
    for temp in b.get_result():
        print(np.shape(temp[0]))
