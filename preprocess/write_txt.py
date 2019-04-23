import os
import pickle


def read_mask(url):
    path = []
    name = []

    for dirpath, dirnames, filenames in os.walk(url):
        for i in range(filenames.__len__()):
            path.append(dirpath)
            name.append(filenames[i])

    return path, name


def load_node(url):
    with open(url, 'rb') as f:
        list_ = pickle.load(f)
    return list_


def save_node(list_, url):
    with open(url, 'wb') as f:
        pickle.dump(list_, f)


def to_pickle():
    dataset = 'DRIVE'
    root = "./temp/" + dataset
    if not os.path.exists(root + "/catalog"):
        os.mkdir(root + '/catalog')

    train_test = ['training', 'test']
    type = ['images', 'manual', 'coarse']

    for i in range(len(train_test)):
        for j in range(len(type)):
            image_url = root + "/" + train_test[i] + "/" + type[j]
            paths, names = read_mask(image_url)
            if type[j] == 'coarse':
                for k in range(len(names)):
                    names[k] = ("shenqi_Prediction_"+str(k)+".png")

            save_url = root + "/catalog/" + train_test[i] + "_" + type[j] + ".pickle"
            save_node(names, save_url)
            list = load_node(save_url)
            print(save_url,list)

to_pickle()
