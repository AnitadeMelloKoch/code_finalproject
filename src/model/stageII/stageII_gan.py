import os

import numpy as np
import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

from src.model.stageI.generator import build_stageI_generator
from src.model.stageII.generator import build_stageII_generator
from src.model.stageII.discriminator import build_stageII_discriminator
from src.model.stageII.adversarial import build_stageII_adversarial
from src.utils.conditioning_augmentation import build_ca_network
from src.utils.discriminator_utils import build_embedding_compressor
from src.utils.training_utils import adversarial_loss, save_image
from src.utils.data_utils import get_image

from tqdm import tqdm

class StageIIGAN():

    def __init__(
        self,
        base_path="./",
        embedding_dim=100,
        generator_lr=0.0002,
        discriminator_lr=0.0002
    ):
        self.path = os.path.join(base_path, 'stageII')
        self.stageI_path = os.path.join(base_path, 'stageI')
        
        self.embedding_dim = embedding_dim
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr

        self.low_image_size = 64
        self.high_image_size = 256
        self.conditioning_dim = 128
        
        self.generator_optimizer = Adam(lr=generator_lr, beta_1=0.5, beta_2=0.999)
        self.discriminator_optimizer = Adam(lr=discriminator_lr, beta_1=0.5, beta_2=0.999)

        self.generatorI = build_stageI_generator()
        self.generatorI.compile(loss='binary_crossentropy', optimizer=self.generator_optimizer)
        self.get_generatorI()
        self.generatorII = build_stageII_generator()
        self.generatorII.compile(loss='binary_crossentropy', optimizer=self.generator_optimizer)

        self.discriminator = build_stageII_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.discriminator_optimizer)

        self.ca_network = build_ca_network()
        self.ca_network.compile(loss='binary_crossentropy', optimizer='Adam')

        self.embedding_compressor = build_embedding_compressor()
        self.embedding_compressor.compile(loss='binary_crossentropy', optimizer='Adam')

        self.adversarial = build_stageII_adversarial(self.generatorI, self.generatorII, self.discriminator)
        self.adversarial.compile(
            loss=['binary_crossentropy', adversarial_loss],
            loss_weights=[1,2],
            optimizer=self.generator_optimizer
        )
        self.board = tf.summary.create_file_writer(
            os.path.join(self.path, 'logs')
        )

    def save(self):
        save_path = os.path.join(self.path, 'weights')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.generatorII.save_weights(os.path.join(save_path, 'gen.h5'))
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

    def get_generatorI(self):
        self.generatorI.load_weights(
            os.path.join(os.path.join(self.stageI_path, "weights", "gen.h5"))
        )

    def train(self,
        x_train_list,
        train_embeds,
        high_test_embeds,
        epochs,
        batch_size=32):

        self.get_generatorI()

        real = np.ones((batch_size, 1), dtype='float')*0.9
        fake = np.zeros((batch_size, 1), dtype='float')*0.1

        num_batches = int(len(x_train_list)/batch_size)
        indices = np.arange(len(x_train_list))

        image_path = os.path.join(self.path, 'test')
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        for epoch in range(epochs):
            print("Epoch: {}".format(epoch))

            gen_loss = []
            dis_loss = []
            dis_gen = []
            dis_loss_true = []
            dis_wrong = []

            np.random.shuffle(indices)
            x_high_train = [x_train_list[idx] for idx in indices]
            high_train_embeds = train_embeds[indices]

            for i in tqdm(range(num_batches)):
                latent_space = np.random.normal(0,1,size=(batch_size, self.embedding_dim))
                idx = np.random.randint(high_train_embeds.shape[1])
                embedding_text = high_train_embeds[i*batch_size:(i+1)*batch_size,idx,:]
                compressed_embedding = self.embedding_compressor.predict_on_batch(embedding_text)
                compressed_embedding = np.reshape(compressed_embedding, (-1,1,1,self.conditioning_dim))
                compressed_embedding = np.tile(compressed_embedding, (1,4,4,1))

                image_files = x_high_train[i*batch_size:(i+1)*batch_size]
                image_batch = []
                for file in image_files:
                    image_batch.append(get_image(file, high_res=True))
                image_batch = np.array(image_batch)

                low_res_fakes, _ = self.generatorI.predict([embedding_text, latent_space])
                high_res_fakes, _ = self.generatorII.predict([embedding_text, low_res_fakes])

                discriminator_loss = self.discriminator.train_on_batch(
                    [image_batch, compressed_embedding],
                    np.reshape(real, (batch_size, 1))
                )

                discriminator_loss_gen = self.discriminator.train_on_batch(
                    [high_res_fakes, compressed_embedding],
                    np.reshape(fake, (batch_size, 1))
                )

                discriminator_loss_fake = self.discriminator.train_on_batch(
                    [image_batch[:(batch_size-1)], compressed_embedding[1:]],
                    np.reshape(fake[1:], (batch_size -1, 1))
                )

                d_loss = 0.5*np.add(
                    discriminator_loss,
                    0.5*np.add(discriminator_loss_gen, discriminator_loss_fake)
                )
                dis_loss.append(d_loss)

                g_loss = self.adversarial.train_on_batch(
                    [embedding_text, latent_space, compressed_embedding],
                    [K.ones((batch_size, 1)),K.ones((batch_size, 256))]
                )
                gen_loss.append(g_loss)
                dis_gen.append(discriminator_loss_gen)
                dis_loss_true.append(discriminator_loss)
                dis_wrong.append(discriminator_loss_fake)

            print("Discriminator Loss: {}".format(np.mean(dis_loss)))
            print("    Generator Loss: {}".format(np.mean(gen_loss)))

            with self.board.as_default():
                tf.summary.scalar('total_discriminator_loss', np.mean(dis_loss), step=epoch)
                tf.summary.scalar('gen_discriminator_loss', np.mean(dis_gen), step=epoch)
                tf.summary.scalar('true_discriminator_loss', np.mean(dis_loss_true), step=epoch)
                tf.summary.scalar('false_discriminator_loss', np.mean(dis_wrong), step=epoch)
                tf.summary.scalar('generator_loss', np.mean(gen_loss), step=epoch)


            if epoch % 5 == 0:
                latent_space = np.random.normal(0,1,size=(batch_size, self.embedding_dim))
                idx = np.random.randint(high_test_embeds.shape[1])
                embedding_batch = high_test_embeds[0:batch_size, idx,:]

                low_fake_images, _ = self.generatorI.predict([embedding_batch, latent_space])
                high_fake_images, _ = self.generatorII.predict([embedding_batch, low_fake_images])

                save_image(high_fake_images[0], 
                    os.path.join(image_path, 'gen_epoch{}_low.png'.format(epoch))
                )

                save_image(high_fake_images[0], 
                    os.path.join(image_path, 'gen_epoch{}_high.png'.format(epoch))
                )

            if epoch % 25 == 0:
                self.save()

        latent_space = np.random.normal(0,1,size=(batch_size, self.embedding_dim))
        idx = np.random.randint(high_test_embeds.shape[1])
        embedding_batch = high_test_embeds[0:batch_size, idx,:]

        low_fake_images, _ = self.generatorI.predict([embedding_batch, latent_space])
        high_fake_images, _ = self.generatorII.predict([embedding_batch, low_fake_images])

        save_image(high_fake_images[0], 
            os.path.join(image_path, 'gen_final_low.png')
        )

        save_image(high_fake_images[0], 
            os.path.join(image_path, 'gen_final_high.png')
        )

        self.save()


