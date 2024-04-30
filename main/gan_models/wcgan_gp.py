from .basic_gan import BasicGAN, BasicGANPipeline
from .cgan import CGAN
from tensorflow import keras
import tensorflow as tf
import numpy as np
from ..preprocessing_utils.general.postprocessing import postprocess_UDP_TCP_flags
from sys import stdout, stderr
from sklearn.preprocessing import OneHotEncoder
from keras import backend
from keras.constraints import Constraint
import logging


class WCGAN_GP(CGAN):
    def __init__(self, discriminator, generator, latent_dim: int, num_y_cols: int):
        super().__init__(discriminator, generator, latent_dim, num_y_cols)
        self.num_y_cols = num_y_cols
        self.g_input_dim = self.latent_dim + self.num_y_cols
        self.gp_weight = 10.0
        self.num_classes = 2 if num_y_cols == 1 else num_y_cols

    def wasserstein_loss(self, y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)

    def gradient_penalty(
        self,
        batch_size,
        real_samples_without_labels,
        real_labels,
        fake_samples_without_labels,
    ):
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        interpolated = (
            alpha * real_samples_without_labels
            + (1 - alpha) * fake_samples_without_labels
        )

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(tf.concat([interpolated, real_labels], axis=1))
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_D(self, X_real, y_real: np.array):
        # & Generate fake data (train D)
        batch_size = tf.shape(X_real)[0]
        # * for CGAN, generate fake labels and concatenate them to the fake samples
        fake_labels = self.generate_fake_labels(batch_size)
        fake_samples_without_labels = self.generate_fake_samples_without_labels(
            batch_size, fake_labels
        )
        # * concatenate the fake labels to the fake samples
        combined_samples_without_labels = tf.concat(
            [fake_samples_without_labels, X_real], axis=0
        )
        combined_labels = tf.concat([fake_labels, y_real], axis=0)
        # * Assemble labels that say "all fake (WGAN: +1) images", then "all real (WGAN: -1) images"
        labels_discriminator = tf.concat(
            [tf.ones((batch_size, 1)), -tf.ones((batch_size, 1))], axis=0
        )
        # * Train Discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(
                tf.concat([combined_samples_without_labels, combined_labels], axis=1)
            )
            d_loss = self.wasserstein_loss(labels_discriminator, predictions)
            gradient_penalty = self.gradient_penalty(
                batch_size, X_real, y_real, fake_samples_without_labels
            )
            d_loss += self.gp_weight * gradient_penalty

        # * Update Discriminator
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )
        return d_loss

    def train_G(self, batch_size: int, y_labels: np.array):
        fake_labels = self.generate_fake_labels(batch_size)
        # * Assemble labels that say "all real images" (WGAN: -1)
        misleading_labels = -tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            fake_samples_without_labels = self.generate_fake_samples_without_labels(
                batch_size, fake_labels
            )
            predictions = self.discriminator(
                tf.concat([fake_samples_without_labels, fake_labels], axis=1)
            )
            g_loss = self.wasserstein_loss(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return g_loss

    def generate_fake_samples_without_labels(self, n_samples, labels):
        noise = tf.random.normal((n_samples, self.latent_dim))
        input_data = tf.concat([noise, labels], axis=1)
        fake_samples = self.generator(input_data)
        return fake_samples

    def generate_fake_labels(self, batch_size) -> tf.Tensor:
        labels = tf.random.uniform(
            (batch_size,), maxval=self.num_classes, dtype=tf.int32
        )
        labels = tf.cast(labels, tf.float32)
        labels = tf.reshape(labels, (-1, 1))
        return labels

    def train_step(self, real_samples):
        # & Unpack the real data.
        X_real = real_samples[:, : -self.num_y_cols]  # first N cols
        y_real = real_samples[:, -self.num_y_cols :]  # last N cols
        batch_size = tf.shape(real_samples)[0]

        # * in WCGAN, update D more times than G
        UPDATE_D_TIMES = 3
        for i in range(UPDATE_D_TIMES):
            d_loss = self.train_D(X_real, y_real)
        g_loss = self.train_G(batch_size, y_real)

        # & Update loss
        # ! Turn this off for now
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }


class WCGAN_GP_pipeline(BasicGANPipeline):
    def __init__(
        self,
        dataset_filename: str,
        decoding_func,
        pipeline_name: str,
        subset=False,
        batch_size: int = 128,
        latent_dim: int = 32,
        use_balanced_dataset: bool = True,
        # * new params
        d_hidden_layer_width: int = 128,
        d_hidden_layer_depth: int = 5,
        g_hidden_layer_width: int = 128,
        g_hidden_layer_depth: int = 5,
        # * for compressed data
        is_load_compressed_data: bool = False,
        compressed_X: np.array = None,
        compressed_y: np.array = None,
        compressed_X_colnames: list = None,
        compressed_X_encoders: dict = None,
        compressed_y_encoder = None,
        # # * no ip
        # is_no_IP: bool = False,
    ) -> None:
        super().__init__(
            dataset_filename=dataset_filename,
            decoding_func=decoding_func,
            pipeline_name=pipeline_name,
            subset=subset,
            batch_size=batch_size,
            latent_dim=latent_dim,
            use_balanced_dataset=use_balanced_dataset,
            d_hidden_layer_width=d_hidden_layer_width,
            d_hidden_layer_depth=d_hidden_layer_depth,
            g_hidden_layer_width=g_hidden_layer_width,
            g_hidden_layer_depth=g_hidden_layer_depth,
            is_load_compressed_data=is_load_compressed_data,
            compressed_X=compressed_X,
            compressed_y=compressed_y,
            compressed_X_colnames=compressed_X_colnames,
            compressed_X_encoders=compressed_X_encoders,
            compressed_y_encoder=compressed_y_encoder,
            # is_no_IP=is_no_IP,
        )
        self.d_hidden_layer_width = d_hidden_layer_width
        self.d_hidden_layer_depth = d_hidden_layer_depth
        self.g_hidden_layer_width = g_hidden_layer_width
        self.g_hidden_layer_depth = g_hidden_layer_depth


    def get_GAN(self):
        return WCGAN_GP(
            discriminator=self.discriminator,
            generator=self.generator,
            latent_dim=self.latent_dim,
            num_y_cols=self.y_cols_len,
        )

    ## * Tuning tips:
    ## *   Set hyperparameters based on literature
    ## *   make it wider first, then try making it deeper
    ## * instead of TSTR-F1, do TSTR-ROC-AUC metric for binary classification task, and ROC-AUC (one-vs-all) for 5 classes
    ## * focus 2 classes

    def get_generator(self):
        logging.info("Using WCGAN-GP Generator.")
        final_layer = self.create_generator_final_layer(self.X_colnames, y_col_num=0)
        # * For CGANs, the y labels are also added to the generator.
        input_shape = self.latent_dim + self.y_cols_len
        # * the output is just to simulate X, since CGANS already know Y.
        output_shape = self.X.shape[1]

        model = keras.models.Sequential()
        model.add(
            keras.layers.InputLayer(input_shape=(input_shape,))
        )
        for _ in range(self.g_hidden_layer_depth):
            model.add(keras.layers.Dense(self.g_hidden_layer_width, activation="relu", kernel_initializer='glorot_uniform'))

        # * consider these two final layers as one layer
        model.add(keras.layers.Dense(output_shape, activation=None))
        model.add(final_layer)
        model.summary(print_fn=logging.info)
        return model

    def get_discriminator(self):
        logging.info("Using WCGAN-GP Discriminator (Critic).")
        input_shape = self.X.shape[1] + self.y.shape[1]

        model = keras.models.Sequential(name="discriminator")
        model.add(
            keras.layers.InputLayer(input_shape=(input_shape,))
        )
        for _ in range(self.d_hidden_layer_depth):
            model.add(keras.layers.Dense(self.d_hidden_layer_width, activation="relu", kernel_initializer='glorot_uniform'))

        model.add(keras.layers.Dense(1, activation=None))  # linear because wasserstein loss
        model.summary(print_fn=logging.info)
        return model

    def generate_samples(self, num_samples: int, **kwargs):
        # * For CGANs, you also need to condition the input.
        # * This is done by passing one-hot encoded labels.
        # * For the generic generate_samples() function,
        # *     we just generate random labels uniformly.

        # * generate random noise for the generator
        latent_space_samples = tf.random.normal((num_samples, self.latent_dim))

        # * generate random labels as a tensor
        num_classes = len(self.y_encoder.classes_)  # 2 for 2 classes
        random_labels = tf.random.uniform(
            (num_samples,), maxval=num_classes, dtype=tf.int32
        )
        random_labels = tf.cast(random_labels, tf.float32)
        random_labels = tf.reshape(random_labels, (-1, 1))

        latent_and_labels = tf.concat((latent_space_samples, random_labels), axis=1)

        # * generate samples using the generator
        generated_samples_X = self.generator(latent_and_labels).numpy()  # G(noise)
        # * postprocess the generated samples
        # ** For labels with binary outputs, round them to 0 or 1
        generated_samples_X = self.postprocess_generated_samples(generated_samples_X)

        # * since the generated samples only generates X, we need to concatenate the labels too.
        generated_samples = np.hstack((generated_samples_X, random_labels))



        return generated_samples
    @staticmethod
    def wasserstein_loss(y_true, y_pred):
        return backend.mean(y_true * y_pred)

    def compile_and_fit_GAN(self, d_learning_rate: float, g_learning_rate: float, d_beta_1: float, g_beta_1:float, epochs: int):
        # beta_1 = 0.0  # according to the paper???
        # beta_2 = 0.9  # according to the paper???
        # beta_1 = 0.9  # default
        # beta_2 = 0.999  # default
        # beta_1 = 0.5  # works better
        beta_2 = 0.999  # default
        self.gan.compile(
            d_optimizer=keras.optimizers.Adam(
                learning_rate=d_learning_rate, beta_1=0.9, beta_2=beta_2
            ),
            g_optimizer=keras.optimizers.Adam(
                learning_rate=g_learning_rate, beta_1=0.5, beta_2=beta_2
            ),
            loss_fn=self.wasserstein_loss,  # ! not used
        )
        self.history = self.gan.fit(
            self.dataset,
            epochs=epochs,
            verbose=0,
        )
        return self.history
