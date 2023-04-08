import numpy as np
import pandas as pd
from ..general.ip_utils import _deanonymise_IP


def scale_ports_N(data: pd.DataFrame) -> np.array:
    d = (
        data.to_numpy().reshape(-1, 1)
        if isinstance(data, pd.Series)
        else data.to_numpy()
    )
    return d / 65535


def _process_N_WGAN_GP_ips(column: pd.Series):
    """IPs to N_WGAN_GP format (i.e. normalise the 4 bytes individually).
    usage:

    Args:
        column (pd.Series): the series.

    Returns:
        _type_: _description_
    """
    return (
        column.apply(_deanonymise_IP)
        .str.split(".", expand=True)
        .to_numpy(dtype=np.int16)
        / 255
    )
