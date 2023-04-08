from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from .preprocessing_utils.general.encode import (
    one_hot_encode_Proto,
    one_hot_encode_TCP_Flags,
    _preprocess_date_first_seen,
    scale_min_max,
)
from .preprocessing_utils.N.encode import scale_ports_N, _process_N_WGAN_GP_ips
from .preprocessing_utils.N.decode import decode_IP_N_WGAN_GP
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler,
    LabelEncoder,
    LabelBinarizer,
)

from .preprocessing_utils.general.ip_utils import _deanonymise_IP


def decode_TCP_flags_one(*six_flags):
    full_flags = "UAPRSF"
    flag = [full_flags[i] if six_flags[i] > 0.5 else "." for i in range(6)]
    return "".join(flag)


def decode_N_WGAN_GP(X, y, y_encoder, X_labels, X_encoders):
    """Given X and y and their encoders, decode the data into traffic flow.
    If y is None, then only the X is decoded; in this case, y and y_encoder are ignored.

    Args:
        X (np.array): X.
        y (np.array | None): y. Set to None to ignore.
        y_encoder (_type_): _description_
        X_labels (list): the list of X_labels.
        X_encoders (_type_): _description_

    Returns:
        _type_: _description_
    """
    is_decode_y = True if y is not None else False
    full_dataset = np.hstack([X, y]) if is_decode_y else X
    y_class_names = [f"y_is_{c}" for c in y_encoder.classes_]
    full_df = pd.DataFrame(
        full_dataset, columns=(X_labels + y_class_names if is_decode_y else X_labels)
    )

    # decode Duration, Bytes and Packets
    Duration_Bytes_Packets = X_encoders["Duration_Bytes_Packets"].inverse_transform(
        full_df[["Duration", "Bytes", "Packets"]]
    )
    full_df["Duration"] = Duration_Bytes_Packets[:, 0]
    full_df["Bytes"] = Duration_Bytes_Packets[:, 1].astype(int)
    full_df["Packets"] = Duration_Bytes_Packets[:, 2].astype(int)

    # decode ports
    full_df[["SrcPt", "DstPt"]] = (full_df[["SrcPt", "DstPt"]] * 65535).astype(int)

    # decode protocols
    proto_names = ["is_" + p for p in X_encoders["Protocols"].categories_[0]]
    full_df["Proto"] = X_encoders["Protocols"].inverse_transform(full_df[proto_names])
    full_df.drop(proto_names, axis=1, inplace=True)

    # decode TCP flags
    flag_names = ["is_URG", "is_ACK", "is_PSH", "is_RES", "is_SYN", "is_FIN"]
    full_df["Flags"] = full_df[flag_names].apply(
        lambda x: decode_TCP_flags_one(*x), axis=1
    )
    full_df.drop(flag_names, axis=1, inplace=True)

    # decode SrcIP
    src_ip_names = ["srcIP1", "srcIP2", "srcIP3", "srcIP4"]
    full_df["SrcIP"] = decode_IP_N_WGAN_GP(full_df[src_ip_names])
    full_df.drop(src_ip_names, axis=1, inplace=True)

    # decode DstIP
    dst_ip_names = ["dstIP1", "dstIP2", "dstIP3", "dstIP4"]
    full_df["DstIP"] = decode_IP_N_WGAN_GP(full_df[dst_ip_names])
    full_df.drop(dst_ip_names, axis=1, inplace=True)

    full_df["day_seconds"] *= 86400
    time_to_str = full_df["day_seconds"].apply(
        lambda x: str(timedelta(seconds=round(x)))
    )
    # decode Date_First_seen
    days_names = X_encoders["Date_first_seen"].categories_[0]
    full_df["Date_first_seen"] = X_encoders["Date_first_seen"].inverse_transform(
        full_df[days_names]
    )

    full_df["Date_first_seen"] = (
        full_df["Date_first_seen"].apply(lambda x: x[3:]) + " " + time_to_str
    )  # x[3:] should be removeprefix("is_")
    full_df.drop(days_names, axis=1, inplace=True)
    full_df.drop("day_seconds", axis=1, inplace=True)

    # decode class
    if is_decode_y:
        full_df["class"] = y_encoder.inverse_transform(
            full_df[y_class_names].to_numpy()
        )

    # rearrange classes
    full_df = full_df[
        [
            "Date_first_seen",
            "Duration",
            "Proto",
            "SrcIP",
            "SrcPt",
            "DstIP",
            "DstPt",
            "Packets",
            "Bytes",
            "Flags",
        ]
        + (["class"] if is_decode_y else [])
    ]
    return full_df


def get_N_WGAN_GP_preprocessed_data(
    data: pd.DataFrame, binary_labels=False, include_date_ip=False
):
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

    dur_bytes_packets_scaler, dur_bytes_packets_scaled = scale_min_max(
        subset[["Duration", "Bytes", "Packets"]]
    )
    ports_normed = scale_ports_N(subset[["SrcPt", "DstPt"]])
    enc_proto, onehotencoded_proto, proto_labels = one_hot_encode_Proto(subset["Proto"])
    onehotencoded_flags, flags_labels = one_hot_encode_TCP_Flags(subset["Flags"])
    full_X = np.hstack(
        [
            dur_bytes_packets_scaled,
            ports_normed,
            onehotencoded_proto,
            onehotencoded_flags,
        ]
    )
    # labels: ["Duration", "Bytes", "Packets", "SrcPt", "DstPt", PROTOCOLS GO HERE, TCP FLAGS GO HERE]
    labels = (
        ["Duration", "Bytes", "Packets", "SrcPt", "DstPt"] + proto_labels + flags_labels
    )
    encoders = {
        "Duration_Bytes_Packets": dur_bytes_packets_scaler,
        "Protocols": enc_proto,
    }

    # when including date and ip, fetch their columns as well (always yes for GANs)
    if include_date_ip:
        src_ips = _process_N_WGAN_GP_ips(subset["SrcIP"])
        dest_ips = _process_N_WGAN_GP_ips(subset["DstIP"])
        (
            enc_date_first_seen,
            onehotencoded_days_of_week,
            days_of_week_labels,
            seconds_normed,
        ) = _preprocess_date_first_seen(subset["Date_first_seen"])
        full_X = np.hstack(
            [full_X, src_ips, dest_ips, onehotencoded_days_of_week, seconds_normed]
        )
        labels += (
            [f"srcIP{i}" for i in range(1, 5)]
            + [f"dstIP{i}" for i in range(1, 5)]
            + days_of_week_labels
            + ["day_seconds"]
        )
        encoders["Date_first_seen"] = enc_date_first_seen

    y, y_encoder = _encode_y(subset["class"].to_numpy())

    return full_X, y, y_encoder, labels, encoders


def get_N_WGAN_GP_preprocessed_dataframe(data, binary_labels=True):
    full_X, y, y_encoder, labels = get_N_WGAN_GP_preprocessed_data(data, binary_labels)
    full_dataset = np.hstack([full_X, y.reshape(-1, 1)])
    full_df = pd.DataFrame(full_dataset, columns=labels + ["class"])
    full_df["class"] = y_encoder.inverse_transform(full_df["class"].astype(int))
    return full_df


def _encode_y(y: pd.Series):
    # y_encoder = LabelEncoder().fit(y)
    y_encoder = LabelBinarizer().fit(y)
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


def get_B_WGAN_GP_preprocessed_data(
    data: pd.DataFrame, binary_labels=False, include_date_ip=False
):
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
    dur_scaler, dur_scaled = scale_min_max(subset["Duration"])
    enc_proto, onehotencoded_proto, proto_labels = one_hot_encode_Proto(subset["Proto"])
    source_ports_binary = _int_to_binary_cols(subset["SrcPt"], 16)
    dest_ports_binary = _int_to_binary_cols(subset["DstPt"], 16)
    bytes_binary = _int_to_binary_cols(subset["Bytes"], 32)
    packets_binary = _int_to_binary_cols(subset["Packets"], 32)
    onehotencoded_flags, flags_labels = one_hot_encode_TCP_Flags(subset["Flags"])

    full_X = np.hstack(
        [
            dur_scaled,
            onehotencoded_proto,
            source_ports_binary,
            dest_ports_binary,
            bytes_binary,
            packets_binary,
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
    encoders = {
        "Duration": dur_scaler,
        "Protocols": enc_proto,
    }

    if include_date_ip:  # when including date and ip, fetch their columns as well
        src_ips = _process_B_WGAN_GP_ips(subset["SrcIP"])
        dest_ips = _process_B_WGAN_GP_ips(subset["DstIP"])
        (
            enc_date_first_seen,
            onehotencoded_days_of_week,
            days_of_week_labels,
            seconds_normed,
        ) = _preprocess_date_first_seen(subset["Date_first_seen"])
        full_X = np.hstack(
            [full_X, src_ips, dest_ips, onehotencoded_days_of_week, seconds_normed]
        )
        labels += (
            [f"0bSrcIP{i+1}" for i in range(32)]
            + [f"0bSrcIP{i+1}" for i in range(32)]
            + days_of_week_labels
            + ["day_seconds"]
        )
        encoders["Date_first_seen"] = enc_date_first_seen

    return full_X, y, y_encoder, labels, encoders


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


_binary_str_to_int = np.vectorize(lambda s: int(s, 2))


def _decode_B_WGAN_GP_ips(thirtytwo_cols):
    four_parts = np.hsplit(thirtytwo_cols, 4)
    combined = [part.astype(str) for part in four_parts]
    combined = [
        np.apply_along_axis(lambda x: "".join(x), 1, part.astype(str))
        for part in combined
    ]
    to_int = np.array([_binary_str_to_int(part) for part in combined], dtype=str).T
    to_ip_str = np.apply_along_axis(lambda x: ".".join(x), 1, to_int)

    return pd.Series(to_ip_str)
