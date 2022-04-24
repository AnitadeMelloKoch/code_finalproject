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

        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.kl_loss = tf.keras.losses.KLDivergence()

        # setting these values as hard because models are hard coded with this
        # maybe set up a config file with these values?
        self.image_size = 64
        self.conditioning_dim = 128
        
        self.generator_optimizer = Adam(lr=generator_lr, beta_1=0.5, beta_2=0.999)
        self.discriminator_optimizer = Adam(lr=discriminator_lr, beta_1=0.5, beta_2=0.999)
        
        self.generator = build_stageI_generator()

        self.discriminator = build_stageI_discriminator()

        self.ca_network = build_ca_network()

        self.embedding_compressor = build_embedding_compressor()

        self.path = os.path.join(base_path, 'stageI')

        self.board = tf.summary.create_file_writer(
            os.path.join(self.path, 'logs')
        )

        self.true_loss = []
        self.incorrect_caption_loss = []
        self.fake_loss = []
        self.dis_loss = []

        self.gen_loss = []

    def save(self):
        save_path = os.path.join(self.path, 'weights')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.generator.save_weights(os.path.join(save_path, 'gen.h5'))
        self.discriminator.save_weights(os.path.join(save_path, 'dis.h5'))
        self.ca_network.save_weights(os.path.join(save_path, 'ca.h5'))
        self.embedding_compressor.save_weights(os.path.join(save_path, 'emb.h5'))

    def load(self):
        save_path = os.path.join(self.path, 'weights')
        if os.path.exists(os.path.join(save_path, 'gen.h5')):
            self.generator.load_weights(os.path.join(save_path, 'gen.h5'))
        if os.path.exists(os.path.join(save_path, 'dis.h5')):
            self.discriminator.load_weights(os.path.join(save_path, 'dis.h5'))
        if os.path.exists(os.path.join(save_path, 'ca.h5')):
            self.ca_network.load_weights(os.path.join(save_path, 'ca.h5'))
        if os.path.exists(os.path.join(save_path, 'emb.h5')):
            self.embedding_compressor.load_weights(os.path.join(save_path, 'emb.h5'))

    def discriminator_loss(self, true_output, incorrect_caption_output, fake_output):
        true_loss = self.loss(tf.ones_like(true_output), true_output)
        incorrect_caption_loss = self.loss(tf.zeros_like(incorrect_caption_output), incorrect_caption_output)
        fake_loss = self.loss(tf.zeros_like(fake_output), fake_output)

        total_loss = true_loss + incorrect_caption_loss + fake_loss

        self.true_loss.append(true_loss)
        self.incorrect_caption_loss.append(incorrect_caption_loss)
        self.fake_loss.append(fake_loss)
        self.dis_loss.append(total_loss)

        return total_loss

    def generator_loss(self, fake_output):
        bin_loss = self.loss(tf.ones_like(fake_output), fake_output)
        kl_loss = self.kl_loss(tf.ones_like(fake_output), fake_output)

        gen_loss = bin_loss + 2*kl_loss

        self.gen_loss.append(gen_loss)

        return gen_loss

    def start_epoch(self):

        self.true_loss = []
        self.incorrect_caption_loss = []
        self.fake_loss = []
        self.dis_loss = []

        self.gen_loss = []

    def end_epoch(self, epoch):
        with self.board.as_default():
            tf.summary.scalar('True_discriminator_loss', np.mean(self.true_loss), step=epoch)
            tf.summary.scalar('Incorrect_caption_discriminator_loss', np.mean(self.incorrect_caption_loss), step=epoch)
            tf.summary.scalar('Fake_discriminator_loss', np.mean(self.fake_loss), step=epoch)
            tf.summary.scalar('Total_discriminator_loss', np.mean(self.dis_loss), step=epoch)
            tf.summary.scalar('Generator_loss', np.mean(self.gen_loss), step=epoch)
            

    def train(self,
        x_train_files,
        test_files,
        train_embeds_list,
        test_embeds,
        end_epoch,
        start_epoch=0,
        batch_size=32):

        num_batches = int(len(x_train_files)/batch_size)
        indices = np.arange(len(x_train_files))

        image_path = os.path.join(self.path, 'test')
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        for epoch in range(start_epoch, end_epoch):
            print("Epoch: {}".format(epoch))

            np.random.shuffle(indices)
            x_train = [x_train_files[idx] for idx in indices]
            train_embeds = train_embeds_list[indices]

            self.start_epoch()

            for i in tqdm(range(num_batches-1)):

                latent_space = np.random.normal(0,1,size=(batch_size, self.embedding_dim))
                idx = np.random.randint(train_embeds.shape[1])
                embedding_text = train_embeds[i*batch_size:(i+1)*batch_size, idx,:]
                compressed_embedding = self.embedding_compressor.predict_on_batch(embedding_text)
                compressed_embedding = np.reshape(compressed_embedding, (-1,1,1,128))
                compressed_embedding = np.tile(compressed_embedding, (1,4,4,1))

                idx = np.random.randint(train_embeds.shape[1])
                embedding_text_wrong = train_embeds[(i+1)*batch_size:(i+2)*batch_size, idx,:]
                compressed_embedding_wrong = self.embedding_compressor.predict_on_batch(embedding_text_wrong)
                compressed_embedding_wrong = np.reshape(compressed_embedding_wrong, (-1,1,1,128))
                compressed_embedding_wrong = np.tile(compressed_embedding_wrong, (1,4,4,1))

                image_files = x_train[i*batch_size:(i+1)*batch_size]
                image_batch = []
                for file in image_files:
                    image_batch.append(get_image(file, high_res=False))
                image_batch = np.array(image_batch)

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                
                    gen_images = self.generator([embedding_text, latent_space], training=True)

                    real_output = self.discriminator([image_batch[:batch_size//2], compressed_embedding[:batch_size//2]], training=True)
                    capt_wrong_output = self.discriminator([image_batch[batch_size-batch_size//2:], compressed_embedding_wrong[batch_size-batch_size//2:]], training=True)
                    gen_output = self.discriminator([gen_images, compressed_embedding], training=True)

                    gen_loss = self.generator_loss(gen_output)
                    disc_loss = self.discriminator_loss(real_output, capt_wrong_output, gen_output)

                gradients_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                gradients_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

                self.generator_optimizer.apply_gradients(zip(gradients_gen, self.generator.trainable_variables))
                self.discriminator_optimizer.apply_gradients(zip(gradients_disc, self.discriminator.trainable_variables))

            print("Discriminator Loss: {}".format(np.mean(self.dis_loss)))
            print("    Generator Loss: {}".format(np.mean(self.gen_loss)))

            self.end_epoch(epoch)
            
            if epoch % 5 == 0:
                latent_space = np.random.normal(0,1,size=(batch_size, self.embedding_dim))
                idx = np.random.randint(train_embeds.shape[1])
                embedding_batch = test_embeds[0:batch_size, idx,:]
                gen_images = self.generator.predict_on_batch(
                    [embedding_batch, latent_space]
                )

                
                save_image(gen_images[0], 
                    os.path.join(image_path, 'gen_epoch{}.png'.format(epoch)),
                    test_files[0]
                )

            self.save()
            
        latent_space = np.random.normal(0,1,size=(batch_size, self.embedding_dim))
        idx = np.random.randint(train_embeds.shape[1])
        embedding_batch = test_embeds[0:batch_size, idx,:]
        gen_images = self.generator.predict_on_batch(
            [embedding_batch, latent_space]
        )
        save_image(gen_images[3], 
            os.path.join(image_path, 'gen_final.png'),
            test_files[3]
        )

        self.save()

