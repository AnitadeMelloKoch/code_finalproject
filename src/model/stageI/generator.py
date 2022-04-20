from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LeakyReLU, ReLU, Activation
from tensorflow.keras.layers import Conv2D, Concatenate, Dense
from tensorflow.keras.layers import Lambda, Reshape

from src.utils.conditioning_augmentation import conditioning_augmentation
from src.utils.generator_utils import UpSamplingBlock
        
def build_stageI_generator():
    input_layer1 = Input(shape=(768,))
    ca = Dense(256)(input_layer1)
    ca = LeakyReLU(alpha=0.2)(ca)

    # get conditioned text
    c = Lambda(conditioning_augmentation)(ca)

    input_layer2 = Input(shape=(100,))
    concat = Concatenate(axis=1)([c, input_layer2])

    x = Dense(16384, use_bias=False)(concat)
    x = ReLU()(x)
    x = Reshape((4,4,1024), input_shape=(16384,))(x)

    x = UpSamplingBlock(x, 512)
    x = UpSamplingBlock(x, 256)
    x = UpSamplingBlock(x, 128)
    x = UpSamplingBlock(x, 64)

    x = Conv2D(
        3, 
        kernel_size=3, 
        padding='same',
        strides=1,
        use_bias=False,
        kernel_initializer='he_uniform' 
    )(x)
    x = Activation('tanh')(x)

    model = Model(inputs=[input_layer1, input_layer2], outputs=[x, ca])

    return model
