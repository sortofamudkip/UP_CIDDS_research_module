from .basic_gan import BasicGAN, BasicGANPipeline


class CGAN(BasicGAN):
    def __init__(self, discriminator, generator, latent_dim: int):
        super().__init__(discriminator, generator, latent_dim)

    # TODO: override this function!!
    def train_step(self, data):
        return super().train_step(data)


class CGAN_pipeline(BasicGANPipeline):
    def __init__(
        self,
        dataset_filename: str,
        decoding_func,
        pipeline_name: str,
        subset=0.25,
        batch_size: int = 128,
    ) -> None:
        super().__init__(
            dataset_filename, decoding_func, pipeline_name, subset, batch_size
        )

    def get_GAN(self):
        return CGAN(
            discriminator=self.discriminator,
            generator=self.generator,
            latent_dim=self.num_cols,
        )
