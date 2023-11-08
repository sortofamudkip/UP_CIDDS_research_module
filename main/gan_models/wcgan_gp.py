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


class WCGAN_GP(CGAN):
    def __init__(self, discriminator, generator, latent_dim: int, num_y_cols: int):
        super().__init__(discriminator, generator, latent_dim)
        self.num_y_cols = num_y_cols
        self.g_input_dim = self.latent_dim + self.num_y_cols
        self.gp_weight = 10.0

    def gradient_penalty(self, batch_size, real_samples, fake_samples):
        # & Get the interpolated samples
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        diff = fake_samples - real_samples
        interpolated = real_samples + alpha * diff

        # & Get the gradients of the interpolated samples
        with tf.GradientTape() as t:
            t.watch(interpolated)
            pred = self.discriminator(interpolated)
        grads = t.gradient(pred, [interpolated])[0]

        # & Calculate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_D(self, real_samples, y_labels: np.array):
        # & Generate fake data (train D)
        batch_size = tf.shape(real_samples)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        # * for CGAN, noise_for_generator will also have the one-hot encoded y labels concatenated.
        noise_for_generator = tf.concat((random_latent_vectors, y_labels), axis=1)

        # & Generate synthetic data
        # ^ generated_images shape: (batch_size, ncols(X))
        generated_images_X = self.generator(noise_for_generator)
        # ^ synthetic_concat_labels shape: (batch_size, ncols(X)+ncols(Y))
        synthetic_concat_labels = tf.concat((generated_images_X, y_labels), axis=1)

        # vstack rows of real and fake data to form the training set's X
        all_samples = tf.concat((real_samples, synthetic_concat_labels), axis=0)
        # * stack rows of real and fake labels to form the training set's y.
        # *   in a WGAN, real is -1, fake is +1
        all_samples_labels = tf.concat(
            [-1 * tf.ones((batch_size, 1)), tf.ones((batch_size, 1))], axis=0
        )

        # get logits of real and fake data
        # * logits_real shape: (batch_size, 1)
        # * logits_fake shape: (batch_size, 1)
        logits_real, logits_fake = tf.split(
            self.discriminator(all_samples), num_or_size_splits=2, axis=0
        )

        # & Train Discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(all_samples)
            d_loss = self.loss_fn(all_samples_labels, predictions)
            gp = self.gradient_penalty(
                batch_size,
                logits_real,  # logits of real samples
                logits_fake,  # logits of fake samples
            )
            d_loss += self.gp_weight * gp
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )
        return d_loss

    def train_G(self, batch_size: int, y_labels: np.array):
        # & Generate fake data (train G)
        random_latent_vectors_X = tf.random.normal(shape=(batch_size, self.latent_dim))
        # * for CGAN, random_vector_labels will also have the one-hot encoded y labels concatenated.
        #       note that this goes BEFORE the latent space is fed into the generator.
        random_vector_labels = tf.concat([random_latent_vectors_X, y_labels], axis=1)

        # & Train Generator
        # Assemble labels that say "all real (WGAN: -1) images", even though all samples here are fake.
        misleading_labels = -1 * tf.ones((batch_size, 1))
        with tf.GradientTape() as tape:
            generated_samples_X = self.generator(random_vector_labels)
            sythetic_concat_labels = tf.concat((generated_samples_X, y_labels), axis=1)
            predictions = self.discriminator(sythetic_concat_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return g_loss

    def train_step(self, real_samples):
        # & Unpack the real data.
        # X_labels = real_samples[:, : -self.num_y_cols]  # first N cols
        y_labels = real_samples[:, -self.num_y_cols :]  # last N cols
        batch_size = tf.shape(real_samples)[0]

        # * in WCGAN, update D more times than G
        UPDATE_D_TIMES = 5
        d_losses = np.zeros(UPDATE_D_TIMES)
        for i in range(UPDATE_D_TIMES):
            d_loss = self.train_D(real_samples, y_labels)
            d_losses[i] = d_loss
        g_loss = self.train_G(batch_size, y_labels)

        # & Update loss
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_losses.mean())
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
        subset=0.25,
        batch_size: int = 128,
        latent_dim: int = 32,
    ) -> None:
        super().__init__(
            dataset_filename,
            decoding_func,
            pipeline_name,
            subset,
            batch_size,
            latent_dim,
        )

    def get_GAN(self):
        return CGAN(
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
        final_layer = self.create_generator_final_layer(self.X_colnames, y_col_num=0)
        # * For CGANs, the y labels are also added to the generator.
        input_shape = self.latent_dim + self.y_cols_len
        # * the output is just to simulate X, since CGANS already know Y.
        output_shape = self.X.shape[1]
        generator = keras.Sequential(
            [
                keras.layers.Dense(80, activation="relu", input_shape=(input_shape,)),
                keras.layers.Dense(80, activation="relu"),
                keras.layers.Dense(80, activation="relu"),
                keras.layers.Dense(80, activation="relu"),
                keras.layers.Dense(80, activation="relu"),
                keras.layers.Dense(output_shape),  # number of features
                final_layer,
            ],
            name="generator",
        )
        return generator

    def get_discriminator(self):
        input_shape = self.X.shape[1] + self.y.shape[1]
        discriminator = keras.Sequential(
            [
                keras.layers.Dense(
                    80,
                    activation="relu",
                    input_shape=(input_shape,),
                ),
                keras.layers.Dense(
                    80,
                    activation="relu",
                ),
                keras.layers.Dense(
                    80,
                    activation="relu",
                ),
                keras.layers.Dense(
                    80,
                    activation="relu",
                ),
                keras.layers.Dense(
                    80,
                    activation="relu",
                ),
                keras.layers.Dense(
                    1, activation="linear"
                ),  # for WGAN, it's linear, not sigmoid
            ],
            name="discriminator",
        )
        return discriminator

    def generate_samples(self, num_samples: int, **kwargs):
        # * For CGANs, you also need to condition the input.
        # * This is done by passing one-hot encoded labels.
        # * For the generic generate_samples() function,
        # *     we just generate random labels uniformly.
        # TODO: In the future, generate_samples() should also be a parameter,
        # TODO:     so we can specify how to generate the samples.

        latent_space_samples = tf.random.normal(
            (num_samples, self.latent_dim)
        )  # (num_samples samples of noise, number of cols)

        # * generate random labels.
        num_classes = len(self.y_encoder.classes_)  # 2 for 2 classes
        rand_labels = np.random.randint(
            low=0, high=num_classes, size=(num_samples)
        ).reshape(-1, 1)
        # * one-hot encode random labels
        enc = OneHotEncoder(
            categories=[list(range(num_classes))],
            dtype="float32",
            sparse=False,
            drop="if_binary",  # ~ this is needed to not generate 2 cols for 2 classes
        ).fit(rand_labels)
        one_hot_labels = tf.convert_to_tensor(enc.transform(rand_labels))
        latent_and_labels = tf.concat((latent_space_samples, one_hot_labels), axis=1)

        # * since the generated samples only generates X, we need to concatenate the labels too.
        generated_samples_X = self.generator(latent_and_labels).numpy()  # G(noise)
        generated_samples = np.hstack((generated_samples_X, one_hot_labels))
        return generated_samples

    # implementation of wasserstein loss in Keras
    @staticmethod
    def wasserstein_loss(y_true, y_pred):
        return backend.mean(y_true * y_pred)

    def compile_and_fit_GAN(self, d_learning_rate, g_learning_rate, beta_1, epochs):
        beta_1 = 0.0  # according to the paper???
        beta_2 = 0.9  # according to the paper???
        # beta_1 = 0.9  # default
        # beta_2 = 0.999  # default
        self.gan.compile(
            d_optimizer=keras.optimizers.Adam(
                learning_rate=d_learning_rate, beta_1=beta_1, beta_2=beta_2
            ),
            g_optimizer=keras.optimizers.Adam(
                learning_rate=g_learning_rate, beta_1=beta_1, beta_2=beta_2
            ),
            loss_fn=self.wasserstein_loss,
        )
        self.history = self.gan.fit(
            self.dataset,
            epochs=epochs,
            verbose=0,
        )
        return self.history
