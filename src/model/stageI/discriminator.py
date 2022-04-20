from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import Flatten, concatenate

from src.utils.discriminator_utils import ConvBlock


def build_stageI_discriminator():
    # takes in 64x64 image from generator and spatially replicated embedding
    input_layer1 = Input(shape=(64,64,3))

    x = Conv2D(
        64, 
        kernel_size=(4,4),
        strides=2,
        padding='same',
        use_bias=False,
        kernel_initializer='he_uniform'
    )(input_layer1)
    x = LeakyReLU(alpha=0.2)(x)
    x = ConvBlock(x, 128)
    x = ConvBlock(x, 256)
    x = ConvBlock(x, 512)

    # compressed and spatially replicated text embedding
    input_layer2 = Input(shape=(4,4,128))
    concat = concatenate([x, input_layer2])

    y = Conv2D(
        512,
        kernel_size=(1,1),
        padding='same',
        strides=1,
        use_bias=False,
        kernel_initializer='he_uniform'
    )(concat)
    y = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(y)
    y = LeakyReLU(alpha=0.2)(y)

    y = Flatten()(y)
    y = Dense(1)(y)
    y = Activation('sigmoid')(y)

    model = Model(inputs=[input_layer1, input_layer2], outputs=[y])

    return model