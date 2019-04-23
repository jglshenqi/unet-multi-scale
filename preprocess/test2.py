from  skimage import filters
import public
import numpy as np

z = 2.3264

data = public.load_hdf5("./temp/STARE/br/temp.hdf5") / 255
print(np.shape(data), np.max(data), np.mean(data))

lthres = filters.threshold_otsu(data)
uthres = data[data > lthres].mean() + (z * data[data > lthres].std())
print(lthres, uthres)
