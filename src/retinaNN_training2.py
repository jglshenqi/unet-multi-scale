from help_functions import load_node, load_hdf5, get_model
import threading
import time
from multiprocessing import Manager
import get_config
import sys
from retinaNN_training import back_calling
import numpy as np

sys.path.append("./src/network/")
config = get_config.get_config()


def process_line(line):
    tmp = [int(val) for val in line.strip().split(',')]
    x = np.array(tmp[:-1])
    y = np.array(tmp[-1:])
    return x, y


def read_files(path, patch_name):
    x = []
    y = []
    for line in patch_name:
        x_url = path + line[0]
        y_url = path + line[1]
        x.append(load_hdf5(x_url))
        y.append(load_hdf5(y_url))
    return x, y


def generate_val_arrays_from_file(path, validation_split, num_of_loss):
    patch_names = load_node(path + "/patch_names.pickle")

    patch_names = patch_names[int(len(patch_names) * validation_split):]
    x, y = read_files(path, patch_names)
    X = np.array(x)
    Y = []
    y = np.array(y)
    y = np.concatenate((1 - y, y), axis=1)
    for i in range(num_of_loss):
        # X.append(x)
        Y.append(np.array(y))
    return X, Y


class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
        self.__flag = threading.Event()
        self.__flag.set()
        self.__running = threading.Event()
        self.__running.set()

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None

    def stop(self):
        self.__flag.set()  # 将线程从暂停状态恢复, 如何已经暂停的话
        self.__running.clear()  # 设置为False


def producer():
    while True:
        np.random.shuffle(patch_names)
        for line in patch_names:
            xy = []
            if q.full():
                while q.qsize() > 2 * config.batch_size:
                    # print("have a break")
                    time.sleep(0.005)
                    break
            # if len(stack)>max_of_q:
            #     while len(stack)>int(max_of_q/10):
            #         break

            x_url = train_patch_url + line[0]
            y_url = train_patch_url + line[1]
            xy.append(load_hdf5(x_url)[0])
            xy.append(load_hdf5(y_url)[0])
            # stop = input()
            q.put(xy)
            # stack.append(xy)


def consumer():
    cal_iter = 0
    while True:
        x = []
        y = []
        X = []
        Y = []
        for i in range(config.batch_size):
            while q.empty():
                time.sleep(0.005)
            xy = q.get()
            # xy = stack.pop()
            x.append(xy[0:1])
            y.append(xy[1:2])
        X = np.array(x)
        y = np.array(y)
        if config.softmax_index == 1:
            y = np.concatenate((1 - y, y), axis=1)
        for i in range(num_of_loss):
            Y.append(y)
        yield X, Y
        cal_iter = cal_iter + 1
        if cal_iter % 100 == 0:
            print(cal_iter / 10)


if __name__ == '__main__':
    print("=====save config for training=====")
    config.save_config()
    config.choose_GPU()
    print("")

    print("\n=====load the network=====")
    save_url = './' + config.name_experiment + '/' + config.name_experiment + '_architecture.json'
    model = get_model((1, 1, config.patch_height, config.patch_width), config, save_url)
    print("")

    # ============  Training ==================================
    print("\n=====start training=====")
    # filepath = './' + config.name_experiment + '/' + config.name_experiment + '_{epoch:02d}_weights.h5'
    filepath = './' + config.name_experiment + '/' + config.name_experiment + '_best_weights.h5'
    call = back_calling(filepath, save_best=True)

    train_patch_url = config.path_data + "/" + "train_patch_" + str(config.patch_width) + "_" + str(config.N_sampling)
    validation_split = 0.1
    num_of_loss = config.num_of_loss
    validation_data = generate_val_arrays_from_file(train_patch_url, validation_split=1 - validation_split,
                                                    num_of_loss=num_of_loss)
    print(validation_data[0].shape)

    max_of_q = 500
    q = Manager().Queue(max_of_q)
    patch_names = load_node(train_patch_url + "/patch_names.pickle")
    patch_names = patch_names[:int(len(patch_names) * (1 - validation_split))]

    a = MyThread(producer)
    a.start()

    b = MyThread(consumer)
    b.start()

    model.fit_generator(generator=b.get_result(),
                        samples_per_epoch=int(config.N_sampling * (1 - validation_split) / config.batch_size),
                        epochs=config.N_epochs,
                        verbose=2,
                        shuffle=True,
                        validation_data=validation_data,
                        workers=1,
                        callbacks=[call.checkpointer, call.history])

    # ========== Save the last model ===================
    model.save_weights('./' + config.name_experiment + '/' + config.name_experiment + '_last_weights.h5',
                       overwrite=True)
