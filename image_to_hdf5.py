import numpy as np
import h5py
import os

import multiprocessing
from scipy.misc import imread
from scipy.misc import imresize

batch_size = 100000
image_size = 128
num_cpus = multiprocessing.cpu_count()

def process(f):
    global image_size
    im = imread(f, mode='RGB')
    im = imresize(im, (image_size, image_size), interp='bicubic')
    return im

## Train
prefix = '/home/ILSVRC2012/train/'
l = list(map(lambda x : os.path.join(prefix, x), os.listdir(prefix)))

i = 0
imagenet = np.zeros((len(l), image_size, image_size, 3), dtype='uint8')
pool = multiprocessing.Pool(num_cpus)
while i < len(l):
    current_batch = l[i:i + batch_size]    
    current_res = np.array(pool.map(process, current_batch))
    imagenet[i:i + batch_size] = current_res    
    i += batch_size
    print(i, 'images')

# Val
prefix = '/home/ILSVRC2012/val/'
l_val = list(map(lambda x : os.path.join(prefix, x), os.listdir(prefix)))

i = 0
imagenet_val = np.zeros((len(l_val), image_size, image_size, 3), dtype='uint8')
pool = multiprocessing.Pool(multiprocessing.cpu_count())

while i < len(l_val):
    current_batch = l_val[i:i + batch_size]    
    current_res = np.array(pool.map(process, current_batch))
    imagenet_val[i:i + batch_size] = current_res    
    i += batch_size
    print(i, 'images')

with h5py.File('/home/aroyer/Datasets/imagenet-128.hdf5', 'w') as f:
    f['train'] = imagenet
    f['val'] = imagenet_val