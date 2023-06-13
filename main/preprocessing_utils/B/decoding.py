import numpy as np
import pandas as pd

_binary_str_to_int = np.vectorize(lambda s: int(s, 2))


def _decode_B_WGAN_GP_ips(thirtytwo_cols):
    four_parts = np.hsplit(thirtytwo_cols.columns, 4)
    binaries = np.array([1 << n for n in range(7, -1, -1)] * 4)
    combined = thirtytwo_cols * binaries
    first = combined[four_parts[0]].sum(axis=1).astype(int).astype(str)
    second = combined[four_parts[1]].sum(axis=1).astype(int).astype(str)
    third = combined[four_parts[2]].sum(axis=1).astype(int).astype(str)
    fourth = combined[four_parts[3]].sum(axis=1).astype(int).astype(str)
    return first + "." + second + "." + third + "." + fourth


# def _decode_B_WGAN_GP_ports(sixteen_cols):
#     four_parts = np.hsplit(sixteen_cols, 4)
#     combined = [part.astype(str) for part in four_parts]
#     combined = [
#         np.apply_along_axis(lambda x: "".join(x), 1, part.astype(str))
#         for part in combined
#     ]
#     to_int = np.array([_binary_str_to_int(part) for part in combined], dtype=str).T
#     to_ip_str = np.apply_along_axis(lambda x: ".".join(x), 1, to_int)

#     return pd.Series(to_ip_str)
