import tensorflow as tf
from tensorflow import keras
from .abstract import GenericPipeline
import pickle
import numpy as np
import pandas as pd
from ..score_dataset import (
    EvaluateSyntheticDataRealisticnessCallback,
    score_data_plausibility_single,
)
from ..preprocess_data import decode_N_WGAN_GP
import wandb
from wandb.keras import WandbMetricsLogger  # , WandbModelCheckpoint


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

    def compile(self, d_optimizer, g_optimizer, loss_fn, custom_metrics=[]):
        if custom_metrics:
            super().compile(metrics=["accuracy"])
        else:
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

        # stack rows of real and fake data to form the training set's X
        all_samples = tf.concat((real_samples, generated_images), axis=0)
        # stack rows of real and fake labels to form the training set's y.
        # real is 1, fake is 0
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
        # Assemble labels that say "all real images".
        misleading_labels = tf.ones((batch_size, 1))
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

    def __init__(
        self, dataset_filename, subset=0.25, batch_size=128, use_wandb=False
    ) -> None:
        super().__init__()
        self.subset = subset
        self.batch_size = batch_size
        self.history = None
        train_loader, num_cols = self.load_data(dataset_filename)
        self.dataset = train_loader
        self.num_cols = num_cols
        self.discriminator = self.get_discriminator()
        self.generator = self.get_generator()
        self.gan = self.get_GAN()
        self.use_wandb = use_wandb
        if use_wandb:
            self.init_wandb()

    def init_wandb(self):
        wandb.init(
            # set the wandb project where this run will be logged
            project="GAN for CIDDS",
            # track hyperparameters and run metadata
            config={
                "architecture": "GAN-basic",
                "preprocessing": "N",
            },
        )

    def load_data(self, filename: str):
        ################################
        #    Loading data from file    #
        ################################
        with open(filename, "rb") as f:
            X, y, y_encoder, X_colnames, X_encoders = pickle.load(f)
            self.X_colnames = X_colnames
            self.y_colnames = [f"y_is_{c}" for c in y_encoder.classes_]
            self.all_col_labels = self.X_colnames + self.y_colnames
            self.X_encoders = X_encoders
            self.y_encoder = y_encoder

        full_dataset = np.hstack([X, y]).astype(np.float32)

        ################################
        #     Subset to save time      #
        ################################
        if self.subset is not False:
            idx = np.random.randint(
                full_dataset.shape[0], size=int(full_dataset.shape[0] * self.subset)
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
        dataset = dataset.shuffle(buffer_size=8192).batch(self.batch_size)
        return dataset, sampled_dataset.shape[1]

    def get_discriminator(self):
        discriminator = keras.Sequential(
            [
                keras.layers.Dense(
                    256, activation="relu", input_shape=(self.num_cols,)
                ),
                keras.layers.Dropout(0.15),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.15),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dropout(0.15),
                keras.layers.Dense(1, activation="sigmoid"),  # is real or fake
            ],
            name="discriminator",
        )
        return discriminator

    def get_generator(self):
        generator = keras.Sequential(
            [
                keras.layers.Dense(64, activation="relu", input_shape=(self.num_cols,)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
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

    def compile_and_fit_GAN(self, learning_rate=0.001, beta_1=0.9, epochs=2):
        self.gan.compile(
            d_optimizer=keras.optimizers.Adam(
                learning_rate=learning_rate, beta_1=beta_1
            ),
            g_optimizer=keras.optimizers.Adam(
                learning_rate=learning_rate, beta_1=beta_1
            ),
            loss_fn=keras.losses.BinaryCrossentropy(),
            custom_metrics=[self.domain_knowledge_score],
        )
        self.history = self.gan.fit(
            self.dataset,
            epochs=epochs,
            callbacks=[
                EvaluateSyntheticDataRealisticnessCallback(
                    self.gan,
                    generate_samples_func=self.generate_samples,
                    num_samples_to_generate=1000,
                    decoder_func=self.decode_samples,
                ),
            ]
            + [WandbMetricsLogger()]
            if self.use_wandb
            else [],
        )
        return self.history

    def domain_knowledge_score(self):
        generated_samples = self.generate_samples(1000)
        decoded = self.decode_samples(generated_samples)
        score = score_data_plausibility_single(decoded)
        return score

    def generate_samples(self, num_samples: int, **kwargs):
        latent_space_samples = tf.random.normal(
            (num_samples, self.num_cols)
        )  # (5 samples of noise, number of cols)
        generated_samples = self.generator(latent_space_samples).numpy()  # G(noise)
        return generated_samples

    def decode_samples(self, generated_samples):
        y_cols_len = len(self.y_encoder.classes_)
        X, y = generated_samples[:, :-y_cols_len], generated_samples[:, -y_cols_len:]
        return decode_N_WGAN_GP(
            X,
            y,
            self.y_encoder,
            self.X_colnames,
            self.X_encoders,
        )
        # fake_y = np.zeros(num_samples).reshape(-1,1) # all 0s since we don't care atm
        # preprocessor.decode_N_WGAN_GP(X=generated_samples, y=fake_y, y_encoder=y_encoder, labels=labels, X_encoders=X_encoders)
