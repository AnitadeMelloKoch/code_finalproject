from tensorflow.keras import Input, Model

def build_stageI_adversarial(generator, discriminator):
    input_layer1 = Input(shape=(768,))
    input_layer2 = Input(shape=(100,))
    input_layer3 = Input(shape=(4,4,128))

    x = generator([input_layer1, input_layer2])

    discriminator.trainable = False

    probabilities = discriminator([x, input_layer3])

    model = Model(inputs=[input_layer1, input_layer2, input_layer3], outputs=[probabilities])

    return model
