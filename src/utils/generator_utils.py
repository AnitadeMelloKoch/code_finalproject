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

def UpSamplingBlock(x, num_kernels):

    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(
        num_kernels,
        kernel_size=(3,3),
        padding='same',
        strides=1,
        use_bias=False,
        kernel_initializer='he_uniform'
    )(x)
    x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
    x = ReLU()(x)

    return x

def concatenate_along_dims(inputs):
    "Join text and images along same dimensions"

    c = inputs[0]
    x = inputs[1]

    c = K.expand_dims(c, axis=1)
    c = K.expand_dims(c, axis=1)
    c = K.tile(c, [1,16,16,1])

    return K.concatenate([c, x], axis=3)