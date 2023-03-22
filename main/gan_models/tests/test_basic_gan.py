from .. import basic_gan


def test_basic_gan_can_be_instantiated():
    basic_gan_pipeline = basic_gan.BasicGANPipeline("./preprocessed/X_y_5_classes_N")
    assert basic_gan_pipeline.num_cols == 31
    assert (
        str(type(basic_gan_pipeline.discriminator))
        == "<class 'keras.engine.sequential.Sequential'>"
    )
    assert (
        str(type(basic_gan_pipeline.generator))
        == "<class 'keras.engine.sequential.Sequential'>"
    )


def test_basic_gan_compile_and_fit():
    basic_gan_pipeline = basic_gan.BasicGANPipeline("./preprocessed/X_y_5_classes_N")
    basic_gan_pipeline.compile_and_fit_GAN()
    assert True
