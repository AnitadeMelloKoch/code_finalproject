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
from tensorflow.keras.regularizers import L2

def ConvBlock(x, num_kernels, kernel_size=(4,4), strides=2, activation=True):
    x = Conv2D(
        num_kernels,
        kernel_size=kernel_size,
        padding='same',
        strides=strides,
        use_bias=False,
        kernel_initializer='he_uniform',
        kernel_regularizer=L2
    )(x)
    x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)

    if activation:
        x = LeakyReLU(alpha=0.2)(x)

    return x

def build_embedding_compressor():
    input_layer = Input(shape=(768,))
    x = Dense(128)(input_layer)
    x = ReLU()(x)

    model = Model(inputs=[input_layer], outputs=[x])
    return model