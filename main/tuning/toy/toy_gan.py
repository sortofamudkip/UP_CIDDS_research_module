import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from hyperparams import DEFAULT_HYPERPARAMS_TO_TUNE, recursive_dict_union
import logging


class WCGAN_GP(tf.keras.Model):
    def __init__(
        self,
        output_dim: int,
        num_classes: int,
        hyperparams_to_tune: dict,  # from get_hyperparams_to_tune() in hyperparams.py
    ):
        super(WCGAN_GP, self).__init__()
        self.latent_dim = 100  # ! fixed for now
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.num_y_cols = 1 if num_classes == 2 else num_classes
        self.gp_weight = 10.0  # ^ probably don't need to tune this

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

        # define the generator model
        model = models.Sequential()
        model.add(
            layers.InputLayer(input_shape=(self.latent_dim + self.num_classes - 1,))
        )
        model.add(layers.Dense(hp_hidden_layer_width, activation="relu"))
        model.add(layers.Dense(hp_hidden_layer_width, activation="relu"))
        model.add(layers.Dense(self.output_dim, activation="tanh"))
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
        model.add(layers.Dense(1, activation=None))
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

    def generate_fake_labels(self, batch_size):
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
