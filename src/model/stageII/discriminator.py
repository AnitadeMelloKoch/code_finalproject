from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dense, Flatten, LeakyReLU, add,
                                     concatenate)
from src.utils.discriminator_utils import ConvBlock


def build_stageII_discriminator():
    input_layer1 = Input(shape=(256,256,3))

    x = Conv2D(
        64,
        kernel_size=(4,4),
        padding='same',
        strides=2,
        use_bias=False,
        kernel_initializer='he_uniform'
    )(input_layer1)
    x = LeakyReLU(alpha=0.2)(x)

    x = ConvBlock(x, 128)
    x = ConvBlock(x, 256)
    x = ConvBlock(x, 512)
    x = ConvBlock(x, 1024)
    x = ConvBlock(x, 2048)
    x = ConvBlock(x, 1024, (1,1), 1)
    x = ConvBlock(x, 512, (1,1), 1, False)

    y = ConvBlock(x, 128, (1,1), 1)
    y = ConvBlock(y, 128, (3,3), 1)
    y = ConvBlock(y, 512, (3,3), 1, False)

    x2 = add([x, y])
    x2 = LeakyReLU(alpha=0.2)(x2)

    # Concatenate compressed and spatially replicated embedding
    input_layer2 = Input(shape=(4,4,128))
    concat = concatenate([x2, input_layer2])

    x3 = Conv2D(
        512,
        kernel_size=(1,1), 
        strides=1,
        padding='same',
        kernel_initializer='he_uniform'
    )(concat)
    x3 = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x3)
    x3 = LeakyReLU(alpha=0.2)(x3)

    # Flatten and add fc
    x3 = Flatten()(x3)
    x3 = Dense(1)(x3)
    x3 = Activation('sigmoid')(x3)

    model = Model(inputs=[input_layer1, input_layer2], outputs=[x3])

    return model






