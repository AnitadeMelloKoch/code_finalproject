from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import random
import time

import numpy as np
import tensorflow as tf

import PIL
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, ReLU, Activation
from tensorflow.keras.layers import UpSampling2D, Conv2D, Concatenate, Dense, concatenate
from tensorflow.keras.layers import Flatten, Lambda, Reshape, ZeroPadding2D, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def adversarial_loss(y_true, y_pred):
    mean = y_pred[:,:128]
    ls = y_pred[:,128:]
    loss = -ls + 0.5*(-1+tf.math.exp(2*ls)+tf.math.square(mean))
    loss = K.mean(loss)

    return loss

def normalize(input_image, real_image):
    input_image = (input_image/127.5) - 1
    real_image = (real_image/127.5) - 1

    return input_image, real_image

def save_image(file, save_path, title):
    min_val = np.min(file)
    if min_val < 0:
        file = file - min_val
        max_val = np.max(file)
        file = file/max_val

    image = plt.figure(num=1, clear=True)
    ax = image.add_subplot(1,1,1)
    ax.imshow(file)
    ax.axis("off")
    ax.set_title(title)
    plt.savefig(save_path)

def save_tensorboard_image(images, board, title, step):

    with board.as_default():
        tf.summary.image(title, images, step=step)