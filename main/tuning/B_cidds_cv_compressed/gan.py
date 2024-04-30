import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from preprocessing_utils.general.postprocessing import (
    postprocess_UDP_TCP_flags,
)
from hyperparams import DEFAULT_HYPERPARAMS_TO_TUNE, recursive_dict_union
import logging
import numpy as np
import pandas as pd
from score_dataset import mask_plausible_rows
from preprocess_data import decode_N_WGAN_GP


class CIDDS_WCGAN_GP(tf.keras.Model):
    def __init__(
        self,
        output_dim: int,
        num_classes: int,
        x_col_labels: list,
        x_encoders: dict,
        decoder_func: callable,  # e.g. decode_N_WGAN_GP
        y_encoder,
        hyperparams_to_tune: dict,  # from get_hyperparams_to_tune() in hyperparams.py
    ):
        super(CIDDS_WCGAN_GP, self).__init__()
        self.hp_latent_dim = -1  # * placeholder value
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.num_y_cols = 1 if num_classes == 2 else num_classes
        self.gp_weight = 10.0  # ^ probably don't need to tune this
        self.x_col_labels = x_col_labels
        self.x_encoders = x_encoders
        self.decoder_func = decoder_func
        self.y_encoder = y_encoder

        # * Hyperarameters to tune (put this before creating G and D models)
        self.hyperparams_to_tune = hyperparams_to_tune
        self.hp_latent_dim = self.hyperparams_to_tune["latent_dim"]
        self.hp_hidden_layer_depth = self.hyperparams_to_tune["hidden_layer_depth"]

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        hp_g_learning_rate = self.hyperparams_to_tune["generator"]["learning_rate"]
        hp_d_learning_rate = self.hyperparams_to_tune["discriminator"]["learning_rate"]
        hp_beta_1_g = self.hyperparams_to_tune["generator"]["beta_1"]
        hp_beta_1_d = self.hyperparams_to_tune["discriminator"]["beta_1"]
        self.g_optimizer = Adam(learning_rate=hp_g_learning_rate, beta_1=hp_beta_1_g)
        self.d_optimizer = Adam(learning_rate=hp_d_learning_rate, beta_1=hp_beta_1_d)

        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

        self.compile_models()

    def determine_generator_activations(
        self, col_labels: list, y_col_num: int
    ) -> np.array:
        """Usage: determine_generator_activations(self.all_col_labels)

        Args:
            col_labels (list): self.all_col_labels
            y_col_num (int): self.y.shape[1]

        Returns:
            np.array: bool array, true = use sigmoid, false = use relu.
        """
        y_indices = list(
            range(len(col_labels) - 1, len(col_labels) - y_col_num - 1, -1)
        )
        x_sigmoid_indices = [
            i
            for i in range(len(col_labels))
            if col_labels[i].startswith("is_") or col_labels[i].startswith("0b")
        ]
        all_sigmoid_indices = np.array(x_sigmoid_indices + y_indices)
        bool_array = np.zeros(len(col_labels), dtype=int)
        bool_array[all_sigmoid_indices] = 1
        return bool_array

    def create_generator_final_layer(self, col_labels: list, y_col_num: int):
        sigmoid_indices = self.determine_generator_activations(col_labels, y_col_num)
        sigmoid_indices = tf.constant(sigmoid_indices, dtype=tf.int32)
        sigmoid_indices = tf.cast(sigmoid_indices, dtype=tf.bool)

        class FinalActivation(keras.layers.Layer):
            def __init__(self):
                super().__init__()

            def call(self, inputs):
                return tf.where(
                    sigmoid_indices,
                    tf.keras.activations.sigmoid(inputs),
                    tf.keras.activations.relu(inputs),
                )

        return FinalActivation()

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

    def build_generator(self):
        # define the hyperparameters to tune
        hp_hidden_layer_width = self.hyperparams_to_tune["generator"][
            "hidden_layer_width"
        ]

        # define the final activation function (specific to this dataset)
        ## * note: y_col_num=0 because CGAN only generates X, and y is specified before generating X
        final_layer = self.create_generator_final_layer(self.x_col_labels, y_col_num=0)
        # define the generator model
        model = models.Sequential()
        model.add(
            layers.InputLayer(input_shape=(self.hp_latent_dim + self.num_classes - 1,))
        )
        for _ in range(self.hp_hidden_layer_depth):
            model.add(layers.Dense(hp_hidden_layer_width, activation="relu"))

        # * consider these two final layers as one layer
        model.add(layers.Dense(self.output_dim, activation=None))
        model.add(final_layer)
        return model

    def build_discriminator(self):
        # define the hyperparameters to tune
        hp_hidden_layer_width = self.hyperparams_to_tune["discriminator"][
            "hidden_layer_width"
        ]

        model = models.Sequential()
        model.add(
            layers.InputLayer(input_shape=(self.output_dim + self.num_classes - 1,))
        )
        for _ in range(self.hp_hidden_layer_depth):
            model.add(layers.Dense(hp_hidden_layer_width, activation="relu"))

        model.add(layers.Dense(1, activation=None))  # linear because wasserstein loss
        return model

    def compile_models(self):
        self.generator.compile(optimizer=self.g_optimizer, loss=self.loss_fn)
        self.discriminator.compile(
            optimizer=self.d_optimizer, loss=self.wasserstein_loss, metrics=["accuracy"]
        )
        logging.info("Compiled generator and discriminator models")
        logging.info("Generator summary:")
        self.generator.summary(print_fn=logging.info)
        logging.info("Discriminator summary:")
        self.discriminator.summary(print_fn=logging.info)

    def generate_fake_samples_without_labels(self, n_samples, labels):
        noise = tf.random.normal((n_samples, self.hp_latent_dim))
        input_data = tf.concat([noise, labels], axis=1)
        fake_samples = self.generator(input_data)
        return fake_samples

    def wasserstein_loss(self, y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)

    def generate_fake_labels(self, batch_size) -> tf.Tensor:
        labels = tf.random.uniform(
            (batch_size,), maxval=self.num_classes, dtype=tf.int32
        )
        labels = tf.cast(labels, tf.float32)
        labels = tf.reshape(labels, (-1, 1))
        return labels

    def decode_samples_to_human_format(
        self,
        generated_samples_X: np.array,
        generated_samples_y: np.array,
    ):
        decoded_data = self.decoder_func(
            generated_samples_X,
            generated_samples_y,
            self.y_encoder,
            self.x_col_labels,
            self.x_encoders,
        )
        # * TCP flags = ..... if UDP
        postprocessed = postprocess_UDP_TCP_flags(decoded_data)
        return postprocessed

    def postprocess_generated_samples(self, generated_data_without_y: np.array):
        logging.info("Postprocessing generated samples...")
        # get dataframe version of generated data
        generated_data_without_y_df = pd.DataFrame(generated_data_without_y, columns=self.x_col_labels, dtype=np.float32)
        # print(f"before postprocessing: {list(zip(self.x_col_labels, generated_data_without_y_df.head(1).values[0]))}")

        protocol_colnames = ['is_ICMP', 'is_TCP', 'is_UDP']
        tcp_colnames = ['is_URG','is_ACK','is_PSH','is_RES','is_SYN','is_FIN']
        day_colnames = ["is_Monday", "is_Tuesday", "is_Wednesday", "is_Thursday", "is_Friday", "is_Saturday", "is_Sunday"]

        generated_data_without_y_df[protocol_colnames] = generated_data_without_y_df[protocol_colnames].apply(lambda row: row == row.max(), axis=1).astype(int)
        generated_data_without_y_df[day_colnames] = generated_data_without_y_df[day_colnames].apply(lambda row: row == row.max(), axis=1).astype(int)
        # * for tcp_colnames, round to 0 or 1
        generated_data_without_y_df[tcp_colnames] = generated_data_without_y_df[tcp_colnames].round()


        # get list of columns that start with 0b
        binary_colnames = [colname for colname in self.x_col_labels if colname.startswith("0b")]
        # for colnames that start with 0b, round to 0 or 1
        generated_data_without_y_df[binary_colnames] = generated_data_without_y_df[binary_colnames].round()
        # return numpy version of generated data
        # print(f"after postprocessing: {list(zip(self.x_col_labels, generated_data_without_y_df.head(1).values[0]))}")
        logging.info("Finished postprocessing generated samples.")
        return generated_data_without_y_df.values.astype(np.float32)


    def generate_n_plausible_samples(self, n_x_rows: int, n_target_rows: int):
        """
        Generates n_target_rows number of plausible samples using the GAN model.

        Args:
            n_target_rows (int): The number of plausible samples to generate.
            n_x_rows (int): The number of rows in the original dataset (X).

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: A 2D numpy array of shape (n_target_rows, n_features) containing the generated plausible samples.
                - numpy.ndarray: A 1D numpy array of shape (n_iterations,) containing the retention scores for each iteration.
        """
        all_plausible_samples = []
        cur_num_rows = 0
        retention_scores = []

        n_target_rows = int(n_target_rows)  # just in case it's a float

        logging.info(f"Generating {n_target_rows} plausible samples...")
        while cur_num_rows < n_target_rows:
            # * generate samples and decode them to human format (pandas df)
            random_labels = self.generate_fake_labels(n_x_rows)
            samples_X = self.generate_fake_samples_without_labels(
                n_x_rows, random_labels
            ).numpy()
            # postprocess the generated samples
            samples_X = self.postprocess_generated_samples(samples_X)
            random_labels = random_labels.numpy()
            samples_X_y = np.hstack((samples_X, random_labels))
            samples_df = self.decode_samples_to_human_format(
                samples_X,
                random_labels,
            )

            # * filter out implausible rows
            filtered_mask = mask_plausible_rows(samples_df, num_classes=2)
            plausible_samples = samples_X_y[filtered_mask]
            # * add to list of plausible samples
            all_plausible_samples.append(plausible_samples)

            # * update retention score
            retention_score = plausible_samples.shape[0] / n_x_rows
            retention_scores.append(retention_score)
            cur_num_rows += plausible_samples.shape[0]

            # * logging
            logging.info(
                f"Generated {plausible_samples.shape[0]} plausible samples (retention score: {retention_score:.2f})"
            )
            if retention_score <= 0.01:
                # print("Generated less than 0.01 plausible samples!")
                logging.warning("Generated less than 0.01 plausible samples!")
                # return a placeholder array filled with NaNs
                return np.full((1, self.output_dim), np.nan), None

        retention_scores = np.array(retention_scores)

        return np.vstack(all_plausible_samples)[:n_target_rows], retention_scores

    def train_step(self, real_samples_with_labels):
        # unpack data (which is just the entire dataset with labels in one big tensor)
        real_samples = real_samples_with_labels[:, : -self.num_y_cols]
        real_labels = real_samples_with_labels[:, -self.num_y_cols :]
        batch_size = tf.shape(real_samples)[0]

        # Train the discriminator n times
        for i in range(3):
            fake_labels = self.generate_fake_labels(batch_size)
            fake_samples_without_labels = self.generate_fake_samples_without_labels(
                batch_size, fake_labels
            )
            # fake and real samples are concatenated together
            combined_samples_without_labels = tf.concat(
                [fake_samples_without_labels, real_samples], axis=0
            )
            combined_labels = tf.concat([fake_labels, real_labels], axis=0)
            labels_discriminator = tf.concat(
                [tf.ones((batch_size, 1)), -tf.ones((batch_size, 1))], axis=0
            )

            with tf.GradientTape() as tape:
                predictions = self.discriminator(
                    tf.concat(
                        [combined_samples_without_labels, combined_labels], axis=1
                    )
                )
                d_loss = self.wasserstein_loss(labels_discriminator, predictions)
                gradient_penalty = self.gradient_penalty(
                    batch_size, real_samples, real_labels, fake_samples_without_labels
                )
                d_loss += self.gp_weight * gradient_penalty

            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )

        # Train the generator
        fake_labels = self.generate_fake_labels(batch_size)
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

        return {"d_loss": d_loss, "g_loss": g_loss}


# create a keras.callbacks.Callback class that stops training if d_loss or g_loss is NaN
class StopTrainingOnNaNCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs["d_loss"] == np.nan or logs["g_loss"] == np.nan:
            logging.warning("d_loss or g_loss is NaN, stopping training")
            self.model.stop_training = True
