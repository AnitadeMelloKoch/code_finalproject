import os
import numpy as np
import glob
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

HIGH_RES_MEAN = 'data/flowers/high_res_mean.npy'
HIGH_RES_STD = 'data/flowers/high_res_std.npy'
LOW_RES_MEAN = 'data/flowers/low_res_mean.npy'
LOW_RES_STD = 'data/flowers/low_res_std.npy'

def get_flowers_high_res():
    files = glob.glob('data/flowers/images/*')
    # files = files[:5000]

    images = []

    for file_name in tqdm(files):
        image = imread(file_name)
        image = resize(image, (64,64))
        images.append(image)

    images = np.array(images)

    mean = np.mean(images, axis=0)
    std = np.std(images, axis=0)

    np.save('data/flowers/low_res_mean.npy', mean)
    np.save('data/flowers/low_res_std.npy', std)

    # images = images-mean
    # images = images/std

def normalize_images(images):
    mean = np.mean(images, axis=0)
    std = np.std(images, axis=0)

    images = images-mean
    images = images/std

    return images

def normalize_image(image, high_res=True):
    if high_res:
        mean = np.load(HIGH_RES_MEAN)
        std = np.load(HIGH_RES_STD)
    else:
        mean = np.load(LOW_RES_MEAN)
        std = np.load(LOW_RES_STD)

    return (image-mean)/std


def get_image(file_path, high_res=True):
    if high_res:
        size = (256,256)
    else:
        size = (64, 64)

    image = imread(file_path)
    image = resize(image, size)

    return normalize_image(image, high_res)
