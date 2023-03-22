from abc import ABC, abstractmethod


class GenericPipeline(ABC):
    @abstractmethod
    def load_data(eslf, filename, subset=0.05, batch_size=32):
        pass

    @abstractmethod
    def get_discriminator(self):
        pass

    @abstractmethod
    def get_GAN(self):
        pass

    @abstractmethod
    def compile_and_run_GAN(self):
        pass

    @abstractmethod
    def generate_samples(self, num_samples, **kwargs):
        pass
