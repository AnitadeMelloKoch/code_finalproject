from tensorflow.keras import Input, Model

def build_stageII_adversarial(generatorI, generatorII, discriminator):
    conditioned_embedding = Input(shape=(768,))
    latent_space = Input(shape=(100,))
    compressed_replication = Input(shape=(4,4,128))

    input_images, ca = generatorI([conditioned_embedding, latent_space])
    discriminator.trainable = False
    generatorI.trainable = False

    images, ca2 = generatorII([conditioned_embedding, input_images])
    probability = discriminator([images, compressed_replication])

    model = Model(
        inputs=[conditioned_embedding, latent_space, compressed_replication],
        outputs=[probability, ca2]
    )

    return model
