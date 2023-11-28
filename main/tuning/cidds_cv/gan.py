import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from hyperparams import DEFAULT_HYPERPARAMS_TO_TUNE, recursive_dict_union
import logging
import numpy as np


class CIDDS_WCGAN_GP(tf.keras.Model):
    def __init__(
        self,
        output_dim: int,
        num_classes: int,
        x_col_labels: list,
        x_encoders: dict,
        hyperparams_to_tune: dict,  # from get_hyperparams_to_tune() in hyperparams.py
    ):
        super(CIDDS_WCGAN_GP, self).__init__()
        self.latent_dim = 100  # ! fixed for now
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.num_y_cols = 1 if num_classes == 2 else num_classes
        self.gp_weight = 10.0  # ^ probably don't need to tune this
        self.x_col_labels = x_col_labels
        self.x_encoders = x_encoders

        # * Hyperarameters to tune (put this before creating G and D models)
        self.hyperparams_to_tune = recursive_dict_union(
            DEFAULT_HYPERPARAMS_TO_TUNE, hyperparams_to_tune
        )

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        hp_g_learning_rate = self.hyperparams_to_tune["generator"]["learning_rate"]
        hp_d_learning_rate = self.hyperparams_to_tune["discriminator"]["learning_rate"]
        self.g_optimizer = Adam(learning_rate=hp_g_learning_rate, beta_1=0.5)
        self.d_optimizer = Adam(learning_rate=hp_d_learning_rate, beta_1=0.5)

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
        print("col_labels", col_labels)
        print("y_col_num", y_col_num)
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
            layers.InputLayer(input_shape=(self.latent_dim + self.num_classes - 1,))
        )
        model.add(layers.Dense(hp_hidden_layer_width, activation="relu"))
        model.add(layers.Dense(hp_hidden_layer_width, activation="relu"))
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
        model.add(layers.Dense(hp_hidden_layer_width, activation="relu"))
        model.add(layers.Dense(hp_hidden_layer_width, activation="relu"))
        model.add(layers.Dense(1, activation=None))  # linear because wasserstein loss
        return model

    def compile_models(self):
        self.generator.compile(optimizer=self.g_optimizer, loss=self.loss_fn)
        self.discriminator.compile(
            optimizer=self.d_optimizer, loss=self.wasserstein_loss, metrics=["accuracy"]
        )

    def generate_fake_samples_without_labels(self, n_samples, labels):
        noise = tf.random.normal((n_samples, self.latent_dim))
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
        # combined_labels = tf.concat([fake_labels, labels], axis=0)
        # print(combined_labels.numpy())

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
