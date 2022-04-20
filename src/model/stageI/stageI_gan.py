import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from src.model.stageI.adversarial import build_stageI_adversarial
from src.model.stageI.discriminator import build_stageI_discriminator
from src.model.stageI.generator import build_stageI_generator
from src.utils.conditioning_augmentation import build_ca_network
from src.utils.discriminator_utils import build_embedding_compressor
from src.utils.training_utils import adversarial_loss, save_image
from tensorflow.keras.optimizers import Adam
from src.utils.data_utils import get_image

from tqdm import tqdm

class StageIGAN():

    def __init__(
        self,
        base_path="./",
        embedding_dim=100,
        generator_lr=0.0002,
        discriminator_lr=0.0002
    ):
        self.embedding_dim = embedding_dim
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr

        # setting these values as hard because models are hard coded with this
        # maybe set up a config file with these values?
        self.image_size = 64
        self.conditioning_dim = 128
        
        self.generator_optimizer = Adam(lr=generator_lr, beta_1=0.5, beta_2=0.999)
        self.discriminator_optimizer = Adam(lr=discriminator_lr, beta_1=0.5, beta_2=0.999)
        
        self.generator = build_stageI_generator()
        self.generator.compile(loss='mse', optimizer=self.generator_optimizer)

        self.discriminator = build_stageI_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.discriminator_optimizer)

        self.ca_network = build_ca_network()
        self.ca_network.compile(loss='binary_crossentropy', optimizer='Adam')

        self.embedding_compressor = build_embedding_compressor()
        self.embedding_compressor.compile(loss='binary_crossentropy', optimizer='Adam')

        self.adversarial = build_stageI_adversarial(self.generator, self.discriminator)
        self.adversarial.compile(loss=['binary_crossentropy', adversarial_loss],
                    loss_weights=[1, 2],
                    optimizer=self.generator_optimizer)

        self.path = os.path.join(base_path, 'stageI')

        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

    def save(self):
        save_path = os.path.join(self.path, 'weights')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.generator.save_weights(os.path.join(save_path, 'gen.h5'))
        self.discriminator.save_weights(os.path.join(save_path, 'dis.h5'))
        self.ca_network.save_weights(os.path.join(save_path, 'ca.h5'))
        self.embedding_compressor.save_weights(os.path.join(save_path, 'emb.h5'))
        self.adversarial.save_weights(os.path.join(save_path, 'adv.h5'))

    def load(self):
        save_path = os.path.join(self.path, 'weights')
        if os.path.exists(os.path.join(save_path, 'gen.h5')):
            self.generator.load_weights(os.path.join(save_path, 'gen.h5'))
        if os.path.exists(os.path.join(save_path, 'dis.h5')):
            self.discriminator.save_weights(os.path.join(save_path, 'dis.h5'))
        if os.path.exists(os.path.join(save_path, 'ca.h5')):
            self.ca_network.save_weights(os.path.join(save_path, 'ca.h5'))
        if os.path.exists(os.path.join(save_path, 'emb.h5')):
            self.embedding_compressor.save_weights(os.path.join(save_path, 'emb.h5'))
        if os.path.exists(os.path.join(save_path, 'adv.h5')):
            self.adversarial.save_weights(os.path.join(save_path, 'adv.h5'))


    def visualize(self):
        self.board = tf.keras.callbacks.Tensorboard(
            log_dir=os.path.join(self.path, 'logs')
        )
        self.board.set_model(self.generator)
        self.board.set_model(self.discriminator)
        self.board.set_model(self.ca_network)
        self.board.set_model(self.embedding_compressor)

    def train(self,
        x_train_files,
        train_embeds_list,
        test_embeds,
        epochs,
        batch_size=32):

        real = np.ones((batch_size, 1), dtype='float')*0.9
        fake = np.zeros((batch_size, 1), dtype='float')*0.1

        num_batches = int(len(x_train_files)/batch_size)
        indices = np.arange(len(x_train_files))

        image_path = os.path.join(self.path, 'test')
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        for epoch in range(epochs):
            print("Epoch: {}".format(epoch))

            gen_loss = []
            dis_loss = []

            np.random.shuffle(indices)
            x_train = [x_train_files[idx] for idx in indices]
            train_embeds = train_embeds_list[indices]

            for i in tqdm(range(num_batches)):

                latent_space = np.random.normal(0,1,size=(batch_size, self.embedding_dim))
                idx = np.random.randint(train_embeds.shape[1])
                embedding_text = train_embeds[i*batch_size:(i+1)*batch_size, idx,:]
                compressed_embedding = self.embedding_compressor.predict_on_batch(embedding_text)
                compressed_embedding = np.reshape(compressed_embedding, (-1,1,1,128))
                compressed_embedding = np.tile(compressed_embedding, (1,4,4,1))

                image_files = x_train[i*batch_size:(i+1)*batch_size]
                image_batch = []
                for file in image_files:
                    image_batch.append(get_image(file, high_res=False))
                image_batch = np.array(image_batch)

                gen_images, _ = self.generator.predict([embedding_text, latent_space])

                discriminator_loss = self.discriminator.train_on_batch(
                    [image_batch, compressed_embedding],
                    np.reshape(real, (batch_size, 1))
                )
                discriminator_loss_gen = self.discriminator.train_on_batch(
                    [gen_images, compressed_embedding],
                    np.reshape(fake, (batch_size, 1))
                )
                discriminator_loss_wrong = self.discriminator.train_on_batch(
                    [gen_images[:batch_size-1], compressed_embedding[1:]],
                    np.reshape(fake[1:], (batch_size-1,1))
                )

                d_loss = 0.5*np.add(
                    discriminator_loss,
                    0.5*np.add(discriminator_loss_gen, discriminator_loss_wrong)
                )
                dis_loss.append(d_loss)

                g_loss = self.adversarial.train_on_batch(
                    [embedding_text, latent_space, compressed_embedding],
                    [K.ones((batch_size, 1))*0.9, K.ones((batch_size, 256))*0.9]
                )

                gen_loss.append(g_loss)

            print("Discriminator Loss: {:.2f}".format(np.mean(dis_loss)))
            print("    Generator Loss: {:.2f}".format(np.mean(gen_loss)))

            if epoch % 5 == 0:
                latent_space = np.random.normal(0,1,size=(batch_size, self.embedding_dim))
                idx = np.random.randint(train_embeds.shape[1])
                embedding_batch = test_embeds[0:batch_size, idx,:]
                gen_images, _ = self.generator.predict_on_batch(
                    [embedding_batch, latent_space]
                )

                for i, image in enumerate(gen_images[:10]):
                    save_image(image, 
                        os.path.join(image_path, 'gen_epoch{}_{}.png'.format(epoch, i))
                    )

            if epoch % 25 == 0:
                self.save()
            
        self.save()

