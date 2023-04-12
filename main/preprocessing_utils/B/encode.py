import numpy as np
import pandas as pd
from ..general.ip_utils import _deanonymise_IP


def _to_binary_str(num: int, max_len=16) -> str:
    return format(num, f"0{max_len}b").zfill(max_len)[-max_len:]


def _int_to_binary_cols(col: pd.Series, max_len=16) -> np.array:
    return (
        col.apply(lambda val: (_to_binary_str(val, max_len)))
        .str.split("", expand=True)
        .loc[:, 1:max_len]
        .to_numpy()
        .astype(int)
    )


def _process_B_WGAN_GP_ips(column: pd.Series):
    """IPs to B_WGAN_GP format (i.e. binarise a.b.c.d individually then concatenate them).

    Args:
        column (pd.Series): the series.

    Returns:
        np.array: 32 columns of IP.
    """
    ip_cols = column.apply(_deanonymise_IP).str.split(".", expand=True)
    ip_cols = ip_cols.astype(int).applymap(lambda x: _to_binary_str(x, 8))
    ip_cols = ip_cols[0] + ip_cols[1] + ip_cols[2] + ip_cols[3]
    ip_cols = ip_cols.str.split("", expand=True).loc[:, 1:32].to_numpy(dtype=int)

    return ip_cols
