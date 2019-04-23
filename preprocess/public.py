import h5py
import pickle


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["image"][()]


# write images into hdf5 files
def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def save_node(list_, url):
    with open(url, 'wb') as f:
        pickle.dump(list_, f)


def load_node(url):
    with open(url, 'rb') as f:
        list_ = pickle.load(f)
    return list_
