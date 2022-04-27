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
from src.utils.training_utils import adversarial_loss, save_image, save_tensorboard_image
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

        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.kl_loss = tf.keras.losses.KLDivergence()

        self.generatorI = build_stageI_generator()
        self.get_generatorI()
        self.generatorII = build_stageII_generator()

        self.discriminator = build_stageII_discriminator()

        self.ca_network = build_ca_network()

        self.embedding_compressor = build_embedding_compressor()

        self.board = tf.summary.create_file_writer(
            os.path.join(self.path, 'logs')
        )
        self.metric = tf.keras.metrics.BinaryAccuracy()

        self.true_loss = []
        self.true_accuracy = []
        self.incorrect_caption_loss = []
        self.incorrect_caption_accuracy = []
        self.fake_loss = []
        self.fake_accuracy = []
        self.dis_loss = []

        self.gen_loss = []

    def save(self):
        save_path = os.path.join(self.path, 'weights')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.generatorII.save_weights(os.path.join(save_path, 'gen.h5'))
        self.discriminator.save_weights(os.path.join(save_path, 'dis.h5'))
        self.ca_network.save_weights(os.path.join(save_path, 'ca.h5'))
        self.embedding_compressor.save_weights(os.path.join(save_path, 'emb.h5'))

    def load(self):
        save_path = os.path.join(self.path, 'weights')
        if os.path.exists(os.path.join(save_path, 'gen.h5')):
            self.generatorII.load_weights(os.path.join(save_path, 'gen.h5'))
        if os.path.exists(os.path.join(save_path, 'dis.h5')):
            self.discriminator.load_weights(os.path.join(save_path, 'dis.h5'))
        if os.path.exists(os.path.join(save_path, 'ca.h5')):
            self.ca_network.load_weights(os.path.join(save_path, 'ca.h5'))
        if os.path.exists(os.path.join(save_path, 'emb.h5')):
            self.embedding_compressor.load_weights(os.path.join(save_path, 'emb.h5'))

    @staticmethod
    def generate_labels(tensor, is_true):
        if is_true:
            return tf.ones_like(tensor)*0.95 + tf.random.uniform(tensor.shape, minval=-0.25, maxval=0.25)
        else:
            return tf.ones_like(tensor)*0.15 + tf.random.uniform(tensor.shape, minval=-0.075, maxval=0.075)



    def discriminator_loss(self, true_output, incorrect_caption_output, fake_output):
        true_loss = self.loss(self.generate_labels(true_output, True), true_output)
        incorrect_caption_loss = self.loss(self.generate_labels(incorrect_caption_output, False), incorrect_caption_output)
        fake_loss = self.loss(self.generate_labels(fake_output, False), fake_output)

        total_loss = 0.5*(true_loss + 0.5*(incorrect_caption_loss + fake_loss))

        self.true_loss.append(true_loss)
        self.incorrect_caption_loss.append(incorrect_caption_loss)
        self.fake_loss.append(fake_loss)
        self.dis_loss.append(total_loss)

        self.metric.update_state(tf.ones_like(true_output), true_output)
        self.true_accuracy.append(self.metric.result().numpy())

        self.metric.update_state(tf.zeros_like(incorrect_caption_output), incorrect_caption_output)
        self.incorrect_caption_accuracy.append(self.metric.result().numpy())
        
        self.metric.update_state(tf.zeros_like(fake_output), fake_output)
        self.fake_accuracy.append(self.metric.result().numpy())

        return total_loss

    def generator_loss(self, fake_output):
        bin_loss = self.loss(self.generate_labels(fake_output, True), fake_output)
        kl_loss = self.kl_loss(self.generate_labels(fake_output, True), fake_output)

        gen_loss = bin_loss + 2*kl_loss

        self.gen_loss.append(gen_loss)

        return gen_loss

    def start_epoch(self):

        self.true_loss = []
        self.true_accuracy = []
        self.incorrect_caption_loss = []
        self.incorrect_caption_accuracy = []
        self.fake_loss = []
        self.fake_accuracy = []
        self.dis_loss = []

        self.gen_loss = []

    def end_epoch(self, epoch):
        with self.board.as_default():
            tf.summary.scalar('discriminator/true_discriminator_loss', np.mean(self.true_loss), step=epoch)
            tf.summary.scalar('discriminator/incorrect_caption_discriminator_loss', np.mean(self.incorrect_caption_loss), step=epoch)
            tf.summary.scalar('discriminator/fake_discriminator_loss', np.mean(self.fake_loss), step=epoch)
            tf.summary.scalar('discriminator/total_discriminator_loss', np.mean(self.dis_loss), step=epoch)
            
            tf.summary.scalar('generator/generator_loss', np.mean(self.gen_loss), step=epoch)

            tf.summary.scalar('accuracy/true', np.mean(self.true_accuracy), step=epoch)
            tf.summary.scalar('accuracy/incorrect_caption', np.mean(self.incorrect_caption_accuracy), step=epoch)
            tf.summary.scalar('accuracy/fake', np.mean(self.fake_accuracy), step=epoch)

            

    def get_generatorI(self):
        self.generatorI.load_weights(
            os.path.join(os.path.join(self.stageI_path, "weights", "gen.h5"))
        )

    def train(self,
        x_train_list,
        test_files,
        train_embeds,
        high_test_embeds,
        end_epoch,
        batch_size=32,
        start_epoch=0):

        self.get_generatorI()

        num_batches = int(len(x_train_list)/batch_size)
        indices = np.arange(len(x_train_list))

        image_path = os.path.join(self.path, 'test')
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        for epoch in range(start_epoch, end_epoch):
            print("Epoch: {}".format(epoch))

            self.start_epoch()

            np.random.shuffle(indices)
            x_high_train = [x_train_list[idx] for idx in indices]
            high_train_embeds = train_embeds[indices]

            for i in tqdm(range(num_batches-1)):
                latent_space = np.random.normal(0,1,size=(batch_size, self.embedding_dim))
                idx = np.random.randint(high_train_embeds.shape[1])
                embedding_text = high_train_embeds[i*batch_size:(i+1)*batch_size,idx,:]
                compressed_embedding = self.embedding_compressor.predict_on_batch(embedding_text)
                compressed_embedding = np.reshape(compressed_embedding, (-1,1,1,self.conditioning_dim))
                compressed_embedding = np.tile(compressed_embedding, (1,4,4,1))

                idx = np.random.randint(high_train_embeds.shape[1])
                embedding_text_wrong = high_train_embeds[(i+1)*batch_size:(i+2)*batch_size, idx,:]
                compressed_embedding_wrong = self.embedding_compressor.predict_on_batch(embedding_text_wrong)
                compressed_embedding_wrong = np.reshape(compressed_embedding_wrong, (-1,1,1,128))
                compressed_embedding_wrong = np.tile(compressed_embedding_wrong, (1,4,4,1))

                image_files = x_high_train[i*batch_size:(i+1)*batch_size]
                image_batch = []
                for file in image_files:
                    image_batch.append(get_image(file, high_res=True))
                image_batch = np.array(image_batch)

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

                    low_res_fakes = self.generatorI([embedding_text, latent_space], training=False)
                    high_res_fakes = self.generatorII([embedding_text, low_res_fakes], training=True)

                    real_output = self.discriminator([image_batch, compressed_embedding], training=True)
                    capt_wrong_output = self.discriminator([image_batch, compressed_embedding_wrong], training=True)
                    gen_output = self.discriminator([high_res_fakes, compressed_embedding], training=True)

                    gen_loss = self.generator_loss(gen_output)
                    disc_loss = self.discriminator_loss(real_output, capt_wrong_output, gen_output)

                gradients_gen = gen_tape.gradient(gen_loss, self.generatorII.trainable_variables)
                gradients_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

                self.generator_optimizer.apply_gradients(zip(gradients_gen, self.generatorII.trainable_variables))
                self.discriminator_optimizer.apply_gradients(zip(gradients_disc, self.discriminator.trainable_variables))

            print("Discriminator Loss: {}".format(np.mean(self.dis_loss)))
            print("    Generator Loss: {}".format(np.mean(self.gen_loss)))

            self.end_epoch(epoch)

            # if epoch % 5 == 0:
            latent_space = np.random.normal(0,1,size=(batch_size, self.embedding_dim))
            idx = np.random.randint(high_test_embeds.shape[1])
            rand_idx = np.arange(high_test_embeds.shape[0])
            sample = np.random.choice(rand_idx, batch_size)
            embedding_batch = high_test_embeds[sample]
            file_names = [test_files[x] for x in sample]
            embedding_batch = embedding_batch[:, idx,:]

            low_fake_images = self.generatorI.predict([embedding_batch, latent_space])
            high_fake_images = self.generatorII.predict([embedding_batch, low_fake_images])

            for x in range(len(low_fake_images)):

                save_image(low_fake_images[0], 
                    os.path.join(image_path, 'gen_epoch{}_low.png'.format(epoch)),
                    file_names[0]
                )
    
                save_image(high_fake_images[0], 
                    os.path.join(image_path, 'gen_epoch{}_high.png'.format(epoch)),
                    file_names[0]
                )

            save_tensorboard_image(low_fake_images, self.board, 'gen low res epoch: {}'.format(epoch), epoch)
            save_tensorboard_image(high_fake_images, self.board, 'gen high res epoch: {}'.format(epoch), epoch)

            self.save()

        latent_space = np.random.normal(0,1,size=(batch_size, self.embedding_dim))
        idx = np.random.randint(high_test_embeds.shape[1])
        rand_idx = np.arange(high_test_embeds.shape[0])
        sample = np.random.choice(rand_idx, batch_size)
        embedding_batch = high_test_embeds[rand_idx]
        file_names = test_files[rand_idx]
        embedding_batch = sample[:, idx,:]

        low_fake_images = self.generatorI.predict([embedding_batch, latent_space])
        high_fake_images = self.generatorII.predict([embedding_batch, low_fake_images])

        save_image(high_fake_images[0], 
            os.path.join(image_path, 'gen_final_low.png'),
            file_names[0]
        )

        save_image(high_fake_images[0], 
            os.path.join(image_path, 'gen_final_high.png'),
            file_names[0]
        )

        self.save()


