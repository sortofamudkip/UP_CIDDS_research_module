from abc import ABC, abstractmethod


class GenericPipeline(ABC):
    @abstractmethod
    def load_data(self, filename: str, subset=0.05, batch_size=32):
        """Loads data from file and returns the TF dataset and number of columns.

        Args:
            filename (str): file name. File should contain X, y, y_encoder, labels, X_encoders.
            subset (float|None, optional): Portion of random samples to use. Set to None to use all. Defaults to 0.05.
            batch_size (int, optional): batch size to use. Defaults to 32.

        Returns:
            tf.data.Dataset: the tf dataset.
            int: number of cols (attributes) of the dataset.
        """
        pass

    @abstractmethod
    def get_discriminator(self):
        """Creates the discriminator D. Will typically use self.num_cols.

        Returns:
            keras.Sequential: the discriminator.
        """
        pass

    @abstractmethod
    def get_generator(self):
        """Creates the generator G. Will typically use self.num_cols.

        Returns:
            keras.Sequential: the generator.
        """
        pass

    @abstractmethod
    def get_GAN(self):
        """Creates a GAN based on self.get_discriminator() and self.get_generator().

        Returns:
            keras.Model: the GAN.
        """
        pass

    @abstractmethod
    def compile_and_fit_GAN(self):
        """Compiles and fits the GAN from self.get_GAN()."""
        pass

    @abstractmethod
    def generate_samples(self, num_samples: int, **kwargs):
        """Generate samples from the trained GAN's generator.

        Args:
            num_samples (int): number of rows to generate.
        """
        pass

    @abstractmethod
    def decode_samples(self):
        """Decodes samples to mimic original (human-readable) form."""
