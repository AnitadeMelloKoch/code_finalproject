from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import random
import time

import numpy as np
import tensorflow as tf

assert tf.__version__.startswith('2')

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

def conditioning_augmentation(x):
    "mean-logsigma converted into text conditioning variable"

    mean = x[:,:128]
    log_sigma = x[:,128:]

    stddev = tf.math.exp(log_sigma)
    epsilon = K.random_normal(shape=K.constant((mean.shape[1],), dtype='int32'))
    c = mean + stddev*epsilon

    return c

def build_ca_network():

    input_layer = Input(shape=(768,))
    mls = Dense(256)(input_layer)
    mls = LeakyReLU(alpha=0.2)(mls)
    ca = Lambda(conditioning_augmentation)(mls)

    return Model(inputs=[input_layer], outputs=[ca])