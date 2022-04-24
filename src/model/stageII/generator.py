from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, ReLU, Activation
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import Lambda, ZeroPadding2D, add

from src.utils.generator_utils import concatenate_along_dims, UpSamplingBlock
from src.utils.conditioning_augmentation import conditioning_augmentation

def residual_block(input):
    "Residual block with plain identity connections"

    x = Conv2D(
        512,
        kernel_size=(3,3),
        padding='same',
        use_bias=False,
        kernel_initializer='he_uniform'
    )(input)
    x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
    x = ReLU()(x)

    x = Conv2D(
        512,
        kernel_size=(3,3),
        padding='same',
        use_bias=False,
        kernel_initializer='he_uniform'
    )(x)
    x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)

    x = add([x, input])
    x = ReLU()(x)

    return x

def build_stageII_generator():
    input_layer1 = Input(shape=(768,))
    input_images = Input(shape=(64,64,3))

    # Conditioning augmentation
    ca = Dense(256)(input_layer1)
    mls = LeakyReLU(alpha=0.2)(ca)
    c = Lambda(conditioning_augmentation)(mls)

    # Downsampling block
    x = ZeroPadding2D(padding=(1,1))(input_images)
    x = Conv2D(
        128,
        kernel_size=(3,3),
        strides=1,
        use_bias=False,
        kernel_initializer='he_uniform'
    )(x)

    x = ReLU()(x)

    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(
        256,
        kernel_size=(4,4),
        strides=2,
        use_bias=False,
        kernel_initializer='he_uniform'
    )(x)
    x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
    x = ReLU()(x)

    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(
        512,
        kernel_size=(4,4),
        strides=2,
        use_bias=False,
        kernel_initializer='he_uniform'
    )(x)
    x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
    x = ReLU()(x)

    # concatenate text conditioning block with encoded image
    concat = concatenate_along_dims([c, x])

    # Residual Blocks
    x = ZeroPadding2D(padding=(1,1))(concat)
    x = Conv2D(
        512,
        kernel_size=(3,3),
        use_bias=False,
        kernel_initializer='he_uniform'
    )(x)
    x = BatchNormalization(gamma_initializer='ones', beta_initializer='zeros')(x)
    x = ReLU()(x)

    x = residual_block(x)
    x = residual_block(x)
    x = residual_block(x)
    x = residual_block(x)
    
    # Upsampling Blocks
    x = UpSamplingBlock(x, 512)
    x = UpSamplingBlock(x, 256)
    x = UpSamplingBlock(x, 128)
    x = UpSamplingBlock(x, 64)

    x = Conv2D(
        3,
        kernel_size=(3,3),
        padding='same',
        use_bias=False,
        kernel_initializer='he_uniform'
    )(x)
    x = Activation('tanh')(x)

    model = Model(inputs=[input_layer1, input_images], outputs=[x])

    return model


















