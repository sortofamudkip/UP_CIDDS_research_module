import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder


from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder


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
    return (
        enc.transform(protocols).toarray(),
        list(enc.get_feature_names_out(["is"])),
    )  # shape: (671241, 4)


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
    labels = ["is_URG", "is_ACK", "is_PSH", "is_RES", "is_SYN", "is_FIN"]
    return return_value, labels


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


def get_N_WGAN_GP_preprocessed_data(data: pd.DataFrame, binary_labels=False):
    """given data, preprocess and return using the N_WGAN_GP method.
    Usage: X, y, y_encoder, labels = get_N_WGAN_GP_preprocessed_data(data)

    Args:
        data (pd.DataFrame): the dataset.
        binary_labels (bool): Whether to use only binary labels or not. Defaults to False.

        If binary_labels=True, the following happens:
        * rows where y="normal" are set to 0.
        * rows where y="attacker" are set to 1.
        * rows where y="victim" are dropped.
        * rows where y="unknown" are dropped.
            * These are flows where CIDDS were not able to determine the maliciousness of connections going to dest port 80 and 443 (HTTP and HTTPS).
        * rows where y="suspicious" are dropped.
            * These are all flows that access ports/services that are not open to the public, i.e. all other flows.

    Returns:
        list: [full_X, y, y_encoder]
    """
    # if binary labels: drop V, U, S
    to_drop_indices = (
        data[
            (data["class"] == "victim")
            | (data["class"] == "unknown")
            | (data["class"] == "suspicious")
        ].index
        if binary_labels
        else []
    )
    subset = data.drop(to_drop_indices)

    minmax_normed = scale_min_max(subset[["Duration", "Bytes", "Packets"]])
    ports_normed = scale_ports(subset[["SrcPt", "DstPt"]])
    onehotencoded_proto, proto_labels = one_hot_encode_Proto(subset["Proto"])
    onehotencoded_flags, flags_labels = one_hot_encode_TCP_Flags(subset["Flags"])
    full_X = np.hstack(
        [minmax_normed, ports_normed, onehotencoded_proto, onehotencoded_flags]
    )
    y, y_encoder = _encode_y(subset["class"].to_numpy())

    # labels: ["Duration", "Bytes", "Packets", "SrcPt", "DstPt", PROTOCOLS GO HERE, TCP FLAGS GO HERE]
    labels = (
        ["Duration", "Bytes", "Packets", "SrcPt", "DstPt"] + proto_labels + flags_labels
    )

    return full_X, y, y_encoder, labels


def get_N_WGAN_GP_preprocessed_dataframe(data, binary_labels=True):
    full_X, y, y_encoder, labels = get_N_WGAN_GP_preprocessed_data(data, binary_labels)
    full_dataset = np.hstack([full_X, y.reshape(-1, 1)])
    full_df = pd.DataFrame(full_dataset, columns=labels + ["class"])
    full_df["class"] = y_encoder.inverse_transform(full_df["class"].astype(int))
    return full_df


def _encode_y(y: pd.Series):
    y_encoder = LabelEncoder().fit(y)
    y = y_encoder.transform(y)  # to original encoding: y_encoder.inverse_transform(y)
    return y, y_encoder


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


def get_B_WGAN_GP_preprocessed_data(data: pd.DataFrame, binary_labels=False):
    """given data, preprocess and return using the E_WGAN_GP method.

    Args:
        data (pd.DataFrame): the dataset.
        binary_labels (bool): Whether to use only binary labels or not. Defaults to False.

        If binary_labels=True, the following happens:
        * rows where y="normal" are set to 0.
        * rows where y="attacker" are set to 1.
        * rows where y="victim" are dropped.
        * rows where y="unknown" are dropped.
            * These are flows where CIDDS were not able to determine the maliciousness of connections going to dest port 80 and 443 (HTTP and HTTPS).
        * rows where y="suspicious" are dropped.
            * These are all flows that access ports/services that are not open to the public, i.e. all other flows.

    Returns:
        list: [full_X, y, y_encoder]
    """
    # if binary labels: drop V, U, S
    to_drop_indices = (
        data[
            (data["class"] == "victim")
            | (data["class"] == "unknown")
            | (data["class"] == "suspicious")
        ].index
        if binary_labels
        else []
    )
    subset = data.drop(to_drop_indices)
    duration_normed = scale_min_max(subset["Duration"])
    onehotencoded_proto, proto_labels = one_hot_encode_Proto(subset["Proto"])
    source_ports_binary = _int_to_binary_cols(subset["SrcPt"], 16)
    dest_ports_binary = _int_to_binary_cols(subset["DstPt"], 16)
    bytes_binary = _int_to_binary_cols(subset["Bytes"], 32)
    packets_binary = _int_to_binary_cols(subset["Packets"], 32)
    onehotencoded_flags, flags_labels = one_hot_encode_TCP_Flags(subset["Flags"])

    full_X = np.hstack(
        [
            duration_normed,
            onehotencoded_proto,
            source_ports_binary,
            dest_ports_binary,
            bytes_binary,
            packets_binary,
            onehotencoded_proto,
            onehotencoded_flags,
        ]
    )
    y, y_encoder = _encode_y(subset["class"].to_numpy())
    labels = (
        ["Duration"]
        + proto_labels
        + [f"0bSrcPt{i+1}" for i in range(16)]
        + [f"0bDstPt{i+1}" for i in range(16)]
        + [f"0bBytes{i+1}" for i in range(32)]
        + [f"0bPackets{i+1}" for i in range(32)]
        + flags_labels
    )
    return full_X, y, y_encoder, labels
