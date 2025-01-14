from .basic_gan import BasicGAN, BasicGANPipeline
from tensorflow import keras
import tensorflow as tf
import numpy as np
from ..preprocessing_utils.general.postprocessing import postprocess_UDP_TCP_flags
from sys import stdout, stderr
from sklearn.preprocessing import OneHotEncoder
import logging
import pickle


class CGANTest(BasicGAN):
    def __init__(self, discriminator, generator, latent_dim: int, num_y_cols: int):
        super().__init__(discriminator, generator, latent_dim)
        self.num_y_cols = num_y_cols
        self.g_input_dim = self.latent_dim + self.num_y_cols

    def train_D(self, real_samples, y_labels: np.array):
        # print(
        #     f"REAL DATA (shape: {real_samples.numpy().shape})",
        #     real_samples.numpy()[:5, :],
        # )
        # & Generate fake data (train D)
        batch_size = tf.shape(real_samples)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        # * for CGAN, noise_for_generator will also have the one-hot encoded y labels concatenated.
        noise_for_generator = tf.concat((random_latent_vectors, y_labels), axis=1)

        # & Generate synthetic data
        # ^ generated_images shape: (batch_size, ncols(X))
        generated_images_X = self.generator(noise_for_generator)
        # ^ images_concat_labels shape: (batch_size, ncols(X)+ncols(Y))
        images_concat_labels = tf.concat((generated_images_X, y_labels), axis=1)

        # vstack rows of real and fake data to form the training set's X
        all_samples = tf.concat((real_samples, images_concat_labels), axis=0)
        # print(f"ALL DATA (shape: {all_samples.numpy().shape})", all_samples.numpy())
        # stack rows of real and fake labels to form the training set's y.
        #   real is 1, fake is 0
        # *  note: for CGAN, we don't need to touch this, since CGAN just involves increasing number of cols (not rows).
        all_samples_labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # print(
        #     f"ALL LABELS (shape: {all_samples_labels.numpy().shape})",
        #     all_samples_labels.numpy(),
        # )

        # & Train Discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(all_samples)
            d_loss = self.loss_fn(all_samples_labels, predictions)
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
        # Assemble labels that say "all real images", even though all samples here are fake.
        misleading_labels = tf.ones((batch_size, 1))
        with tf.GradientTape() as tape:
            generated_samples_X = self.generator(random_vector_labels)
            images_concat_labels = tf.concat((generated_samples_X, y_labels), axis=1)
            predictions = self.discriminator(images_concat_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return g_loss

    def train_step(self, real_samples):
        # & Unpack the real data.
        # X_labels = real_samples[:, : -self.num_y_cols]  # first N cols
        y_labels = real_samples[:, -self.num_y_cols :]  # last N cols
        batch_size = tf.shape(real_samples)[0]

        d_loss = self.train_D(real_samples, y_labels)
        g_loss = self.train_G(batch_size, y_labels)

        # & Update loss
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }


class CGANTest_pipeline(BasicGANPipeline):
    def __init__(
        self,
        dataset_filename: str,
        decoding_func,
        pipeline_name: str,
        subset=0.25,
        batch_size: int = 128,
        latent_dim: int = 0,
        use_balanced_dataset: bool = True,
    ) -> None:
        super().__init__(
            dataset_filename,
            decoding_func,
            pipeline_name,
            subset,
            batch_size,
            latent_dim,
            use_balanced_dataset,
        )

    def get_GAN(self):
        return CGANTest(
            discriminator=self.discriminator,
            generator=self.generator,
            latent_dim=self.latent_dim,
            num_y_cols=self.y_cols_len,
        )

    def load_data(self, filename: str, use_balanced_dataset: bool):
        ################################
        #    Loading data from file    #
        ################################
        with open(filename, "rb") as f:
            X, y, y_encoder, X_colnames, X_encoders = pickle.load(f)
            self.X = X
            self.y = y
            # * TEST: try very clearly separating values of attacker and normals
            logging.info("Using test X!!!")
            unraveled_y = y.ravel()
            # attacks
            self.X = self.X[:, :2]  # only keep first two cols
            self.X[unraveled_y == 0] = 0.2 + np.random.uniform(
                -0.01, 0.01, size=self.X[unraveled_y == 0].shape
            )
            # normals
            self.X[unraveled_y == 1] = 0.8 + np.random.uniform(
                -0.01, 0.01, size=self.X[unraveled_y == 1].shape
            )
            # logging.debug(self.X[unraveled_y == 1].mean(axis=1).mean())
            # assert False, "test"
            self.X_colnames = X_colnames
            self.y_colnames = (  # if y is binary labels, y.shape[1] == 1
                [f"y_is_{c}" for c in y_encoder.classes_]
                if len(y_encoder.classes_) != 2
                else ["y"]
            )
            self.all_col_labels = self.X_colnames + self.y_colnames
            self.X_encoders = X_encoders
            self.y_encoder = y_encoder

        full_dataset = np.hstack([self.X, self.y]).astype(np.float32)
        # print("full dataset:", full_dataset[:2, :])

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
        if len(y_encoder.classes_) == 2 and use_balanced_dataset:
            logging.info("Using balanced dataset for training.")
            dataset = self.get_balanced_dataset(sampled_dataset)
        else:
            logging.info("Using imbalanced (original) dataset for training.")
            dataset = self.get_unbalanced_dataset(sampled_dataset)
        dataset = dataset.shuffle(buffer_size=self.shuffle_size).batch(self.batch_size)
        return dataset, sampled_dataset.shape[1]

    ## * Tuning tips:
    ## *   Set hyperparameters based on literature
    ## *   make it wider first, then try making it deeper
    ## * instead of TSTR-F1, do TSTR-ROC-AUC metric for binary classification task, and ROC-AUC (one-vs-all) for 5 classes
    ## * focus 2 classes

    def get_generator(self):
        logging.debug("using cgan TEST generator")
        final_layer = self.create_generator_final_layer(self.X_colnames, y_col_num=0)
        # * For CGANs, the y labels are also added to the generator.
        input_shape = self.latent_dim + self.y_cols_len
        # * the output is just to simulate X, since CGANS already know Y.
        output_shape = self.X.shape[1]
        generator = keras.Sequential(
            [
                keras.layers.Dense(32, activation="relu", input_shape=(input_shape,)),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(output_shape),  # number of features
                # final_layer,
            ],
            name="generator",
        )
        return generator

    def get_discriminator(self):
        logging.debug("using cgan TEST discriminator")
        input_shape = self.X.shape[1] + self.y.shape[1]
        discriminator = keras.Sequential(
            [
                keras.layers.Dense(64, activation="relu", input_shape=(input_shape,)),
                # keras.layers.Dropout(0.15),
                keras.layers.Dense(32, activation="relu"),
                # keras.layers.Dropout(0.15),
                keras.layers.Dense(32, activation="relu"),
                # keras.layers.Dropout(0.15),
                keras.layers.Dense(1, activation="sigmoid"),  # is real or fake
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
