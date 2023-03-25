import tensorflow as tf
from tensorflow import keras
from .abstract import GenericPipeline
import pickle
import numpy as np
import pandas as pd
from ..score_dataset import EvaluateSyntheticDataRealisticnessCallback
from ..preprocess_data import decode_N_WGAN_GP


class BasicGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim: int):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim  # number of columns of the dataset
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data.
        real_samples = data  # _ is because we don't care about the labels yet
        ################################
        # Generate fake data (train D) #
        ################################
        batch_size = tf.shape(real_samples)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        noise_for_generator = random_latent_vectors
        # noise_for_generator = tf.concat( # this is not needed until CGAN
        #     [random_latent_vectors, one_hot_labels], axis=1
        # )
        # Generate synthetic data
        generated_images = self.generator(noise_for_generator)
        # print(real_samples.shape, generated_images.shape)
        # stack real and fake data to form the training set's X
        # print(real_samples.numpy())
        # print(generated_images.numpy())
        all_samples = tf.concat((real_samples, generated_images), axis=0)
        # stack real and fake labels to form the training set's y
        all_samples_labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        ################################
        #     Train Discriminator      #
        ################################
        with tf.GradientTape() as tape:
            predictions = self.discriminator(all_samples)
            d_loss = self.loss_fn(all_samples_labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        ################################
        # Generate fake data (train G) #
        ################################
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = random_latent_vectors
        # random_vector_labels = tf.concat(
        #     [random_latent_vectors, one_hot_labels], axis=1
        # )

        ################################
        #       Train Generator        #
        ################################
        ## if D's output (prob) is close to 1, then D thinks that generated_samples is REAL data, i.e. G is doing a good job
        ## if D's output (prob) is close to 0, then D thinks that generated_samples is FAKE data, i.e. G is doing a bad job
        ## if D did well (G did badly), G's loss will be high, so we need to update it a lot
        ## if D did badly (G did well), G's loss will be low, so we don't need to update it that much
        misleading_labels = tf.zeros(
            (batch_size, 1)
        )  # Assemble labels that say "all real images".
        with tf.GradientTape() as tape:
            generated_samples = self.generator(random_vector_labels)
            predictions = self.discriminator(generated_samples)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }


class BasicGANPipeline(GenericPipeline):
    """A basic GAN that only generates unlabled flows (NOT labled).
    Typical usage: basic_gan_pipeline = BasicGANPipeline(), then basic_gan_pipeline.compile_and_fit_GAN()

    Args:
        GenericPipeline (_type_): _description_
    """

    def __init__(self, dataset_filename) -> None:
        super().__init__()
        train_loader, num_cols = self.load_data(dataset_filename)
        self.dataset = train_loader
        self.num_cols = num_cols
        self.discriminator = self.get_discriminator()
        self.generator = self.get_generator()
        self.gan = self.get_GAN()

    def load_data(self, filename: str, subset=0.05, batch_size=32):
        ################################
        #    Loading data from file    #
        ################################
        with open(filename, "rb") as f:
            X, y, y_encoder, labels, X_encoders = pickle.load(f)
            self.dataset_columns = labels
            self.X_encoders = X_encoders

        # these two lines should be in the CGAN
        # y_enc = OneHotEncoder()
        # y_onehot = y_enc.fit_transform(y.reshape(-1,1)).todense()
        full_dataset = X
        full_dataset = full_dataset.astype(np.float32)

        ################################
        #     Subset to save time      #
        ################################
        if subset is not False:
            idx = np.random.randint(
                full_dataset.shape[0], size=int(full_dataset.shape[0] * subset)
            )
            sampled_dataset = full_dataset[idx, :]
        else:
            sampled_dataset = full_dataset

        ################################
        # Convert to Dataloader format #
        ################################
        # it's 0 because we don't care about the label yet
        # train_set = [(sampled_dataset[i], 0) for i in range(sampled_dataset.shape[0])] # to make it in the torch.utils.data.DataLoader format

        # dataset = tf.data.Dataset.from_tensor_slices( (sampled_dataset, np.zeros(sampled_dataset.shape[0]) ))
        dataset = tf.data.Dataset.from_tensor_slices(sampled_dataset)
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
        return dataset, sampled_dataset.shape[1]

    def get_discriminator(self):
        discriminator = keras.Sequential(
            [
                keras.layers.Dense(
                    256, activation="relu", input_shape=(self.num_cols,)
                ),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(1, activation="sigmoid"),  # is real or fake
            ],
            name="discriminator",
        )
        return discriminator

    def get_generator(self):
        generator = keras.Sequential(
            [
                keras.layers.Dense(16, activation="relu", input_shape=(self.num_cols,)),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(self.num_cols),  # number of features
            ],
            name="generator",
        )
        return generator

    def get_GAN(self):
        gan = BasicGAN(
            discriminator=self.discriminator,
            generator=self.generator,
            latent_dim=self.num_cols,
        )
        return gan

    def compile_and_fit_GAN(self, learning_rate=0.0003, epochs=2):
        self.gan.compile(
            d_optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            g_optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss_fn=keras.losses.BinaryCrossentropy(),
        )
        self.gan.fit(
            self.dataset,
            epochs=epochs,
            callbacks=[
                EvaluateSyntheticDataRealisticnessCallback(
                    self.gan,
                    generate_samples_func=self.generate_samples,
                    num_samples_to_generate=100,
                    decoder_func=self.decode_samples,
                )
            ],
        )

    def generate_samples(self, num_samples: int, **kwargs):
        latent_space_samples = tf.random.normal(
            (num_samples, self.num_cols)
        )  # (5 samples of noise, number of cols)
        generated_samples = self.generator(latent_space_samples).numpy()  # G(noise)
        return generated_samples

    def decode_samples(self, generated_samples):
        return decode_N_WGAN_GP(
            generated_samples,
            None,
            None,
            self.dataset_columns,
            self.X_encoders,
        )
        # fake_y = np.zeros(num_samples).reshape(-1,1) # all 0s since we don't care atm
        # preprocessor.decode_N_WGAN_GP(X=generated_samples, y=fake_y, y_encoder=y_encoder, labels=labels, X_encoders=X_encoders)
