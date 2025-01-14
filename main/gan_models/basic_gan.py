import tensorflow as tf
from tensorflow import keras
from .abstract import GenericPipeline
import pickle
import numpy as np
import pandas as pd
from ..score_dataset import (
    EvaluateSyntheticDataRealisticnessCallback,
    score_data_plausibility_single,
    mask_plausible_rows,
)
import logging
from ..preprocessing_utils.general.postprocessing import postprocess_UDP_TCP_flags

# from ..preprocess_data import decode_N_WGAN_GP
from pathlib import Path
import logging


class BasicGAN(keras.Model):
    def __init__(
        self, discriminator: keras.Model, generator: keras.Model, latent_dim: int
    ):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim  # ! unrelated to ncol(X) and ncol(Y)
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn, custom_metrics=[]):
        if custom_metrics:
            super().compile(metrics=["accuracy"])
        else:
            super().compile(run_eagerly=True)
        logging.info("Compiling GAN...")
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.discriminator.compile(optimizer=self.d_optimizer, loss=self.loss_fn)
        self.generator.compile(optimizer=self.g_optimizer, loss=self.loss_fn)
        self.gen_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="g_loss")

    def train_step(self, data):
        # Unpack the real data.
        # #TODO: for CGANs,
        #        we PROBABLY don't need to add another set of one-hot encoded y labels,
        #        since the data already has both X and Y.
        #        or maybe it makes it better? who knows. Have to do more research.
        real_samples = data
        ################################
        # Generate fake data (train D) #
        ################################
        batch_size = tf.shape(real_samples)[0]
        # note: self.latent_dim is numcols(X) + numcols(Y), i.e. one actual row.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        # note: for CGAN, noise_for_generator will also have the one-hot encoded y labels concatenated.
        #       note that this goes BEFORE the latent space is fed into the generator.
        noise_for_generator = random_latent_vectors

        # Generate synthetic data
        generated_images = self.generator(noise_for_generator)

        # stack rows of real and fake data to form the training set's X
        all_samples = tf.concat((real_samples, generated_images), axis=0)
        # stack rows of real and fake labels to form the training set's y.
        #   real is 1, fake is 0
        #   note: for CGAN, we don't need to touch this, since CGAN just involves increasing number of cols (not rows).
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
        # note: for CGAN, noise_for_generator will also have the one-hot encoded y labels concatenated.
        #       note that this goes BEFORE the latent space is fed into the generator.
        random_vector_labels = random_latent_vectors
        # random_vector_labels = tf.concat(
        #     [random_latent_vectors, one_hot_labels], axis=1
        # )

        ################################
        #       Train Generator        #
        ################################
        # Assemble labels that say "all real images", even though all samples here are fake.
        misleading_labels = tf.ones((batch_size, 1))
        with tf.GradientTape() as tape:
            generated_samples = self.generator(random_vector_labels)
            # TODO: for CGAN, the discriminator also takes the one-hot encoded Y as well.
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
    """A basic GAN that only generates unlabled flows (NOT labeled).
    Typical usage: basic_gan_pipeline = BasicGANPipeline(), then basic_gan_pipeline.compile_and_fit_GAN()
    """
    def __init__(
        self,
        dataset_filename: str,  # real dataset filename
        decoding_func,  # currently either decode_N_WGAN_GP or decode_B_WGAN_GP
        pipeline_name: str,  # name of the pipeline, for saving models/datasets.
        subset=0.25,
        batch_size: int = 128,
        latent_dim: int = 0,
        use_balanced_dataset: bool = False,  # whether to balance the dataset or not
        d_hidden_layer_width: int = 128,
        d_hidden_layer_depth: int = 5,
        g_hidden_layer_width: int = 128,
        g_hidden_layer_depth: int = 5,
        # load using compressed data
        is_load_compressed_data: bool = False,
        compressed_X: np.array = None,
        compressed_y: np.array = None,
        compressed_X_colnames: list = None,
        compressed_X_encoders: dict = None,
        compressed_y_encoder = None,
        # # * param if no IP
        # is_no_IP: bool = False,
    ) -> None:
        super().__init__()
        self.subset = subset
        self.batch_size = batch_size
        self.history = None
        self.decoding_func = decoding_func
        self.pipeline_name = pipeline_name
        self.use_balanced_dataset = use_balanced_dataset
        self.shuffle_size = 16384

        # ! these aren't used yet in basic_gan, but are usede in wcgan_gp
        self.d_hidden_layer_width = d_hidden_layer_width
        self.d_hidden_layer_depth = d_hidden_layer_depth
        self.g_hidden_layer_width = g_hidden_layer_width
        self.g_hidden_layer_depth = g_hidden_layer_depth

        # * for compressed data
        self.is_load_compressed_data = is_load_compressed_data
        self.compressed_X = compressed_X
        self.compressed_y = compressed_y
        self.compressed_X_colnames = compressed_X_colnames
        self.compressed_X_encoders = compressed_X_encoders
        self.compressed_y_encoder = compressed_y_encoder

        # # * for no IP
        # self.is_no_IP = is_no_IP

        # these are to be filled out by self.load_data
        self.X = None  # * original (real) X, NOT synthetic
        self.y = None  # * original (real) y, NOT synthetic
        self.X_colnames = None
        self.y_colnames = None
        self.all_col_labels = None
        self.X_encoders = None
        self.y_encoder = None
        train_loader, num_cols = self.load_data(
            dataset_filename, self.use_balanced_dataset
        ) if not is_load_compressed_data else self.load_data_from_compressed_file(
            self.compressed_X, self.compressed_y, self.compressed_X_colnames, self.compressed_X_encoders, self.compressed_y_encoder, self.use_balanced_dataset
            )
        
        # log shapes of X and X_colnames
        logging.info(f"🟠self.X.shape: {self.X.shape}")
        logging.info(f"🟠self.X_colnames shape: {len(self.X_colnames)}")

        self.y_cols_len = self.get_y_cols_len()
        # ^ if latent_dim is not defined, set it to num_cols.
        self.latent_dim = latent_dim if latent_dim != 0 else num_cols

        self.dataset = train_loader
        self.num_cols = num_cols
        self.discriminator = self.get_discriminator()
        self.generator = self.get_generator()
        self.gan = self.get_GAN()

    def create_results_dir(self):
        output_dir = Path(__file__).parent / "../../../results" / self.pipeline_name
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            logging.info(f"Created output dir {output_dir}.")
        except FileExistsError:
            logging.error(f"Dir already exists: {output_dir} ")
            assert False

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
            
            def get_config(self):
                base_config = super().get_config()
                config = {}
                return {**base_config, **config}
            
            @classmethod
            def from_config(cls, config):
                sublayer_config = config.pop("sublayer")
                sublayer = keras.saving.deserialize_keras_object(sublayer_config)
                return cls(sublayer, **config)


        return FinalActivation()

    def postprocess_generated_samples(self, generated_data_without_y: np.array):
        # get dataframe version of generated data
        generated_data_without_y_df = pd.DataFrame(generated_data_without_y, columns=self.X_colnames, dtype=np.float32)
        # print(f"before postprocessing: {list(zip(self.X_colnames, generated_data_without_y_df.head(1).values[0]))}")

        protocol_colnames = ['is_ICMP', 'is_TCP', 'is_UDP']
        tcp_colnames = ['is_URG','is_ACK','is_PSH','is_RES','is_SYN','is_FIN']
        day_colnames = ["is_Monday", "is_Tuesday", "is_Wednesday", "is_Thursday", "is_Friday", "is_Saturday", "is_Sunday"]

        generated_data_without_y_df[protocol_colnames] = generated_data_without_y_df[protocol_colnames].apply(lambda row: row == row.max(), axis=1).astype(int)
        generated_data_without_y_df[day_colnames] = generated_data_without_y_df[day_colnames].apply(lambda row: row == row.max(), axis=1).astype(int)
        # * for tcp_colnames, round to 0 or 1
        generated_data_without_y_df[tcp_colnames] = generated_data_without_y_df[tcp_colnames].round()


        # get list of columns that start with 0b
        binary_colnames = [colname for colname in self.X_colnames if colname.startswith("0b")]
        # for colnames that start with 0b, round to 0 or 1
        generated_data_without_y_df[binary_colnames] = generated_data_without_y_df[binary_colnames].round()
        # return numpy version of generated data
        # print(f"after postprocessing: {list(zip(self.X_colnames, generated_data_without_y_df.head(1).values[0]))}")
        return generated_data_without_y_df.values.astype(np.float32)


    def load_data(self, filename: str, use_balanced_dataset: bool):
        ################################
        #    Loading data from file    #
        ################################
        with open(filename, "rb") as f:
            X, y, y_encoder, X_colnames, X_encoders = pickle.load(f)

            # * for no IP
            if self.is_no_IP:
                logging.info("Removing IP columns from X and X_colnames...")
                # get the indices of all elements starting with "dstIP" and "srcIP"
                dstIP_indices = [
                    i for i in range(len(X_colnames)) 
                    if (
                        ("dstip" in X_colnames[i].lower())
                        or ("srcip" in X_colnames[i].lower())
                    )
                ]
                # drop all columns in X and X_labels with indices in ip_indices
                X = np.delete(X, dstIP_indices, axis=1)
                X_colnames = np.delete(X_colnames, dstIP_indices, axis=0)

            self.X = X
            self.y = y
            self.X_colnames = X_colnames
            self.y_colnames = (  # if y is binary labels, y.shape[1] == 1
                [f"y_is_{c}" for c in y_encoder.classes_]
                if len(y_encoder.classes_) != 2
                else ["y"]
            )
            self.all_col_labels = self.X_colnames + self.y_colnames
            self.X_encoders = X_encoders
            self.y_encoder = y_encoder
        # replace X_encoders with "X_encoders" from filename's directory
        X_encoders_fname = Path(filename).parent / "X_encoders"
        with open(X_encoders_fname, "rb") as f:
            self.X_encoders = pickle.load(f)
            logging.info(f"Loaded X_encoders from {X_encoders_fname}.")

        # replace X_encoders with "X_encoders" from filename's directory
        X_encoders_fname = Path(filename).parent / "X_encoders"
        with open(X_encoders_fname, "rb") as f:
            self.X_encoders = pickle.load(f)
            logging.info(f"Loaded X_encoders from {X_encoders_fname}.")

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
        if len(y_encoder.classes_) == 2 and use_balanced_dataset:
            logging.info("Using balanced dataset for training.")
            dataset = self.get_balanced_dataset(sampled_dataset)
        else:
            logging.info("Using imbalanced (original) dataset for training.")
            dataset = self.get_unbalanced_dataset(sampled_dataset)
        dataset = dataset.shuffle(buffer_size=self.shuffle_size).batch(self.batch_size)
        return dataset, sampled_dataset.shape[1]

    def get_balanced_dataset(self, dataset_np: np.array) -> tf.data.Dataset:
        """
        Returns a balanced dataset as a TensorFlow dataset object. Binary classification only.

        Args:
        - dataset_np: A numpy array containing the dataset (both X and y).

        Returns:
        - A TensorFlow dataset object containing the balanced dataset.
        """
        raveled_y = self.y.ravel()  # by this point, self.y is already initialized
        # split into real and fake
        normal_samples = dataset_np[raveled_y == 1, :]  # 1 is normal
        attacker_samples = dataset_np[raveled_y == 0, :]  # 0 is attacker
        # upsample attacker samples
        num_normal_samples = normal_samples.shape[0]
        num_attacker_samples = attacker_samples.shape[0]
        num_normal_to_attacker_ratio = int(num_normal_samples / num_attacker_samples)
        
        # turn into td.data.Dataset
        normal_dataset = tf.data.Dataset.from_tensor_slices(normal_samples)
        attacker_dataset = tf.data.Dataset.from_tensor_slices(attacker_samples.repeat(num_normal_to_attacker_ratio, axis=0))
        # use tf.data.Dataset.sample_from_datasets to sample from both datasets
        dataset = tf.data.Dataset.sample_from_datasets(
            [normal_dataset, attacker_dataset], [0.5, 0.5]
        )
        return dataset


    def load_data_from_compressed_file(
        self,
        compressed_X : np.array, 
        compressed_y : np.array, 
        compressed_X_colnames : list, 
        compressed_X_encoders : dict, 
        compressed_y_encoder, 
        use_balanced_dataset: bool,
    ):
        
        # # * for no IP
        # if self.is_no_IP:
        #     logging.info("Removing IP columns from X and X_colnames...")
        #     # get the indices of all elements starting with "dstIP" and "srcIP"
        #     dstIP_indices = [
        #         i for i in range(len(compressed_X_colnames)) 
        #         if (
        #             ("dstip" in compressed_X_colnames[i].lower())
        #             or ("srcip" in compressed_X_colnames[i].lower())
        #         )
        #     ]
        #     # drop all columns in X and X_labels with indices in ip_indices
        #     compressed_X = np.delete(compressed_X, dstIP_indices, axis=1)
        #     compressed_X_colnames = np.delete(compressed_X_colnames, dstIP_indices, axis=0)

        self.X = compressed_X
        self.y = compressed_y
        self.X_colnames = list(compressed_X_colnames)
        self.y_colnames = list(  # if y is binary labels, y.shape[1] == 1
                [f"y_is_{c}" for c in compressed_y_encoder.classes_]
                if len(compressed_y_encoder.classes_) != 2
                else ["y"]
            )

        self.all_col_labels = self.X_colnames + self.y_colnames
        logging.info(f"🟠self.all_col_labels: {self.all_col_labels}")
        self.X_encoders = compressed_X_encoders
        logging.info(f"🟠self.X_encoders: {self.X_encoders}")
        self.y_encoder = compressed_y_encoder
        logging.info(f"🟠self.y_encoder: {self.y_encoder}")
        # * create train dataset
        X_y_train = np.hstack([self.X, self.y]).astype(np.float32)
        logging.info(f"🟠X_y_train.shape: {X_y_train.shape}")
        # Create a TensorFlow Dataset for training data
        train_dataset = self.get_balanced_dataset(X_y_train) if use_balanced_dataset else self.get_unbalanced_dataset(X_y_train)
        train_dataset = train_dataset.shuffle(buffer_size=self.shuffle_size).batch(self.batch_size)
        return train_dataset, self.X.shape[1]

    def get_unbalanced_dataset(self, dataset_np: np.array) -> tf.data.Dataset:
        """
        Returns an unbalanced dataset as a TensorFlow dataset object.

        Args:
        - dataset_np: A numpy array containing the dataset.

        Returns:
        - A TensorFlow dataset object containing the unbalanced dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices(dataset_np)
        return dataset

    def get_X_y(self):
        return self.X, self.y_encoder.inverse_transform(self.y)

    def get_discriminator(self):
        discriminator = keras.Sequential(
            [
                keras.layers.Dense(
                    256, activation="relu", input_shape=(self.num_cols,)
                ),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),  # is real or fake
            ],
            name="discriminator",
        )
        return discriminator

    def get_generator(self):
        final_layer = self.create_generator_final_layer(
            self.all_col_labels, self.y.shape[1]
        )
        generator = keras.Sequential(
            [
                keras.layers.Dense(
                    64, activation="relu", input_shape=(self.latent_dim,)
                ),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(self.num_cols),  # number of features
                final_layer,
            ],
            name="generator",
        )
        return generator

    def get_GAN(self):
        return BasicGAN(
            discriminator=self.discriminator,
            generator=self.generator,
            latent_dim=self.latent_dim,
        )
    
    def postprocess_generated_samples(self, generated_data_without_y: np.array):
        # get dataframe version of generated data
        generated_data_without_y_df = pd.DataFrame(generated_data_without_y, columns=self.X_colnames, dtype=np.float32)
        # print(f"before postprocessing: {list(zip(self.X_colnames, generated_data_without_y_df.head(1).values[0]))}")

        protocol_colnames = ['is_ICMP', 'is_TCP', 'is_UDP']
        tcp_colnames = ['is_URG','is_ACK','is_PSH','is_RES','is_SYN','is_FIN']
        day_colnames = ["is_Monday", "is_Tuesday", "is_Wednesday", "is_Thursday", "is_Friday", "is_Saturday", "is_Sunday"]

        generated_data_without_y_df[protocol_colnames] = generated_data_without_y_df[protocol_colnames].apply(lambda row: row == row.max(), axis=1).astype(int)
        generated_data_without_y_df[day_colnames] = generated_data_without_y_df[day_colnames].apply(lambda row: row == row.max(), axis=1).astype(int)
        # * for tcp_colnames, round to 0 or 1
        generated_data_without_y_df[tcp_colnames] = generated_data_without_y_df[tcp_colnames].round()


        # get list of columns that start with 0b
        binary_colnames = [colname for colname in self.X_colnames if colname.startswith("0b")]
        # for colnames that start with 0b, round to 0 or 1
        generated_data_without_y_df[binary_colnames] = generated_data_without_y_df[binary_colnames].round()
        # return numpy version of generated data
        # print(f"after postprocessing: {list(zip(self.X_colnames, generated_data_without_y_df.head(1).values[0]))}")
        return generated_data_without_y_df.values.astype(np.float32)

    def compile_and_fit_GAN(
        self, d_learning_rate: float, g_learning_rate: float, d_beta_1: float, g_beta_1:float, epochs: int
    ):
        self.gan.compile(
            d_optimizer=keras.optimizers.Adam(
                learning_rate=d_learning_rate, beta_1=d_beta_1
            ),
            g_optimizer=keras.optimizers.Adam(
                learning_rate=g_learning_rate, beta_1=g_beta_1
            ),
            loss_fn=keras.losses.BinaryCrossentropy(),
            # custom_metrics=[self.domain_knowledge_score],
            # run_eagerly=True,
        )
        self.history = self.gan.fit(
            self.dataset,
            epochs=epochs,
            # don't generate samples every epoch anymore
            # callbacks=[
            #     EvaluateSyntheticDataRealisticnessCallback(
            #         self.gan,
            #         generate_samples_func=self.generate_samples,
            #         num_samples_to_generate=int(self.X.shape[0] * 0.8),
            #         decoder_func=self.decode_samples_to_human_format,
            #         pipeline_name=self.pipeline_name,
            #     ),
            # ],
            verbose=2,
        )
        return self.history

    def domain_knowledge_score(self):
        generated_samples = self.generate_samples(1000)
        decoded = self.decode_samples(generated_samples)
        score = score_data_plausibility_single(decoded)
        return score

    def generate_samples(self, num_samples: int, **kwargs) -> np.array:
        """
        Generates a given number of samples using the generator of the GAN model.

        Args:
            num_samples (int): The number of samples to generate.
            **kwargs: Additional keyword arguments.

        Returns:
            np.array: Numpy array of generated samples.
        """
        latent_space_samples = tf.random.normal(
            (num_samples, self.latent_dim)
        )  # (num_samples samples of noise, number of cols)
        generated_samples = self.generator(latent_space_samples).numpy()  # G(noise)
        return generated_samples

    def generate_n_plausible_samples(self, n_target_rows: int):
        """
        Generates n_target_rows number of plausible samples using the GAN model.

        Args:
            n_target_rows (int): The number of plausible samples to generate.

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
            samples_np = self.generate_samples(self.X.shape[0])
            samples_df = self.decode_samples_to_human_format(
                samples_np
            )  # ! FOR DEBUG: TURN THIS OFF

            # * filter out implausible rows
            filtered_mask = mask_plausible_rows(
                samples_df, num_classes=2
            )  # ! FOR DEBUG: TURN THIS OFF
            # logging.warning("turning off plausible rows mask!!!")
            # logging.debug(filtered_mask)
            # logging.debug(filtered_mask.shape)
            # plausible_samples = samples_np # & FOR DEBUG: TURN THIS ON
            plausible_samples = samples_np[filtered_mask]  # ! FOR DEBUG: TURN THIS OFF

            # * add to list of plausible samples
            all_plausible_samples.append(plausible_samples)

            # * update retention score
            retention_score = plausible_samples.shape[0] / self.X.shape[0]
            retention_scores.append(retention_score)
            cur_num_rows += plausible_samples.shape[0]

            # * logging
            logging.info(
                f"Generated {plausible_samples.shape[0]} plausible samples (retention score: {retention_score:.2f})"
            )
            if retention_score <= 0.01:
                print("Generated less than 0.01 plausible samples!")
                logging.warning("Generated less than 0.01 plausible samples!")
                assert (
                    False
                ), "Early stopping: generated less than 0.01 plausible samples!"

        retention_scores = np.array(retention_scores)

        return np.vstack(all_plausible_samples)[:n_target_rows], retention_scores

    def decode_samples_to_human_format(self, generated_samples):
        y_cols_len = self.get_y_cols_len()

        X, y = generated_samples[:, :-y_cols_len], generated_samples[:, -y_cols_len:]
        decoded_data = self.decoding_func(
            X,
            y,
            self.y_encoder,
            self.X_colnames,
            self.X_encoders,
        )
        postprocessed = postprocess_UDP_TCP_flags(decoded_data)
        return postprocessed
        # fake_y = np.zeros(num_samples).reshape(-1,1) # all 0s since we don't care atm
        # preprocessor.decode_N_WGAN_GP(X=generated_samples, y=fake_y, y_encoder=y_encoder, labels=labels, X_encoders=X_encoders)

    def decode_samples_to_model_format(self, generated_samples):
        y_cols_len = self.get_y_cols_len()

        X, y = generated_samples[:, :-y_cols_len], generated_samples[:, -y_cols_len:]
        return X, y.ravel(), self.y_encoder

    def get_y_cols_len(self) -> int:
        # for whatever reason, if there are only 2 classes,
        # the number of columns in y is only 1 (instead of one-hot-encoded as 2).
        # so if there are only 2 classes, we only take the last col as y.
        if len(self.y_encoder.classes_) == 2:
            y_cols_len = 1
        else:
            y_cols_len = len(self.y_encoder.classes_)
        return y_cols_len
