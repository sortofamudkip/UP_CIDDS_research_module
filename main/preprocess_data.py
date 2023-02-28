import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def one_hot_encode_Proto(data: pd.Series) -> np.array:
    """one-hot encode the Proto column.
    Usage: one_hot_encode_Proto(data["Proto"])

    Args:
        data (pd.Series): The series as indexed by data["Proto"].

    Returns:
        np.array: np.array of shape (num_rows, 4).
    """
    protocols = data.to_numpy().reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(protocols)
    # print("categories:", enc.categories_)
    return enc.transform(protocols).toarray()  # shape: (671241, 4)


def one_hot_encode_TCP_Flags(data: pd.Series) -> np.array:
    """one-hot encode the Flags column.
    Usage: one_hot_encode_TCP_Flags(data["Flags"])

    Args:
        data (pd.Series): _description_

    Returns:
        np.array: np.array of shape (num_rows, 4).
    """
    return_value = (
        data.str.split("", expand=True)
        .loc[:, 1:6]
        .applymap(lambda x: 0 if x == "." else 1)
        .to_numpy()
    )
    return return_value


def scale_min_max(data: pd.DataFrame) -> np.array:
    d = (
        data.to_numpy().reshape(-1, 1)
        if isinstance(data, pd.Series)
        else data.to_numpy()
    )
    return MinMaxScaler().fit_transform(d)


def scale_ports(data: pd.DataFrame) -> np.array:
    d = (
        data.to_numpy().reshape(-1, 1)
        if isinstance(data, pd.Series)
        else data.to_numpy()
    )
    return d / 65535


def get_N_WGAN_GP_preprocessed_data(data: pd.DataFrame):
    minmax_normed = scale_min_max(data[["Duration", "Bytes", "Packets"]])
    ports_normed = scale_ports(data[["SrcPt", "DstPt"]])
    onehotencoded_proto = one_hot_encode_Proto(data["Proto"])
    onehotencoded_flags = one_hot_encode_TCP_Flags(data["Flags"])
    full_X = np.hstack(
        [minmax_normed, ports_normed, onehotencoded_proto, onehotencoded_flags]
    )
    y = data["class"].to_numpy().reshape(-1, 1)
    return full_X, y
