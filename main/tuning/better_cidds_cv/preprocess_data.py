from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from preprocessing_utils.general.encode import (
    one_hot_encode_Proto,
    one_hot_encode_TCP_Flags,
    _preprocess_date_first_seen,
    scale_min_max,
    _decode_TCP_flags_one,
    _encode_y,
)
from preprocessing_utils.N.encode import scale_ports_N, _process_N_WGAN_GP_ips
from preprocessing_utils.N.decode import decode_IP_N_WGAN_GP

# from preprocessing_utils.general.ip_utils import IP_Preprocessor
from preprocessing_utils.B.encode import (
    _int_to_binary_cols,
    _to_binary_str,
    _process_B_WGAN_GP_ips,
)
import logging


def decode_N_WGAN_GP(
    X, y, y_encoder, X_labels, X_encoders, is_decode_y=True, is_decode_IP_dates=False
) -> pd.DataFrame:
    """Given X and y and their encoders, decode the data into traffic flow.
    If y is None, then only the X is decoded; in this case, y and y_encoder are ignored.

    Args:
        X (np.array): X.
        y (np.array | None): y. Set to None to ignore.
        y_encoder (_type_): y encoders.
        X_labels (list): the list of X_labels.
        X_encoders (dict): X encoders.

    Returns:
        pd.DataFrame: a Dataframe.
    """
    is_decode_y = True if y is not None else False
    full_dataset = np.hstack([X, y]) if is_decode_y else X

    if len(y_encoder.classes_) == 2:
        y_class_names = ["y"]
    else:
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
        lambda x: _decode_TCP_flags_one(*x), axis=1
    )
    full_df.drop(flag_names, axis=1, inplace=True)

    if is_decode_IP_dates:
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

    # reorder
    full_df = full_df[
        [
            "Duration",
            "Proto",
            "SrcPt",
            "DstPt",
            "Packets",
            "Bytes",
            "Flags",
        ]
        + (["Date_first_seen", "SrcIP", "DstIP"] if is_decode_IP_dates else [])
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
        list: [full_X, y, y_encoder, x_labels, x_encoders]
    """
    logging.debug("Start preprocessing data using N_WGAN_GP method.")
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

    logging.debug("Processing cols: Duration, Bytes and Packets...")
    dur_bytes_packets_scaler, dur_bytes_packets_scaled = scale_min_max(
        subset[["Duration", "Bytes", "Packets"]]
    )

    logging.debug("Processing cols: SrcPt and DstPt...")
    ports_normed = scale_ports_N(subset[["SrcPt", "DstPt"]])

    logging.debug("Processing cols: Proto...")
    enc_proto, onehotencoded_proto, proto_labels = one_hot_encode_Proto(subset["Proto"])

    logging.debug("Processing cols: Flags...")
    onehotencoded_flags, flags_labels = one_hot_encode_TCP_Flags(subset["Flags"])

    logging.debug("Assembling full_X...")
    full_X = np.hstack(
        [
            dur_bytes_packets_scaled,
            ports_normed,
            onehotencoded_proto,
            onehotencoded_flags,
        ]
    )

    logging.debug("Assembling x_labels and x_encoders...")
    # labels: ["Duration", "Bytes", "Packets", "SrcPt", "DstPt", PROTOCOLS GO HERE, TCP FLAGS GO HERE]
    x_labels = (
        ["Duration", "Bytes", "Packets", "SrcPt", "DstPt"] + proto_labels + flags_labels
    )
    x_encoders = {
        "Duration_Bytes_Packets": dur_bytes_packets_scaler,
        "Protocols": enc_proto,
    }

    # when including date and ip, fetch their columns as well (always yes for GANs)
    if include_date_ip:
        logging.debug("Processing cols: SrcIP and DstIP...")
        src_ips = _process_N_WGAN_GP_ips(subset["SrcIP"])
        dest_ips = _process_N_WGAN_GP_ips(subset["DstIP"])

        logging.debug("Processing col: Date_first_seen...")
        (
            enc_date_first_seen,
            onehotencoded_days_of_week,
            days_of_week_labels,
            seconds_normed,
        ) = _preprocess_date_first_seen(subset["Date_first_seen"])

        logging.debug("Assembling X data with IP and Date_first_seen...")
        full_X = np.hstack(
            [full_X, src_ips, dest_ips, onehotencoded_days_of_week, seconds_normed]
        )
        x_labels += (
            [f"srcIP{i}" for i in range(1, 5)]
            + [f"dstIP{i}" for i in range(1, 5)]
            + days_of_week_labels
            + ["day_seconds"]
        )
        x_encoders["Date_first_seen"] = enc_date_first_seen

    logging.debug("Processing col: class...")
    y, y_encoder = _encode_y(subset["class"].to_numpy())

    logging.info("Finished preprocessing data using N_WGAN_GP method.")
    return full_X, y, y_encoder, x_labels, x_encoders


def get_preprocessed_dataframe(full_X, y, y_encoder, x_labels):
    full_X = pd.DataFrame(full_X, columns=x_labels)
    y_labels = (
        [f"y_{label}" for label in y_encoder.classes_]
        if len(y_encoder.classes_) != 2
        else ["y"]
    )
    full_y = pd.DataFrame(y, columns=y_labels)
    full_df = pd.concat([full_X, full_y], axis=1)
    return full_df


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
        list: [full_X, y, y_encoder, x_labels, x_encoders]
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
    x_labels = (
        ["Duration"]
        + proto_labels
        + [f"0bSrcPt{i+1}" for i in range(16)]
        + [f"0bDstPt{i+1}" for i in range(16)]
        + [f"0bBytes{i+1}" for i in range(32)]
        + [f"0bPackets{i+1}" for i in range(32)]
        + flags_labels
    )
    x_encoders = {
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
        x_labels += (
            [f"0bSrcIP{i+1}" for i in range(32)]
            + [f"0bDstIP{i+1}" for i in range(32)]
            + days_of_week_labels
            + ["day_seconds"]
        )
        x_encoders["Date_first_seen"] = enc_date_first_seen

    return full_X, y, y_encoder, x_labels, x_encoders
