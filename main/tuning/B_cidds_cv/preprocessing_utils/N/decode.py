import numpy as np


def decode_IP_N_WGAN_GP(four_cols):
    temp = (four_cols * 255).astype(int).astype(str)
    return temp.agg(lambda x: ".".join(x), axis=1)
