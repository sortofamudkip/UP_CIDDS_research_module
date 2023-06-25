from datetime import timedelta
from UP_CIDDS_research_module.main.preprocessing_utils.general.encode import (
    _decode_TCP_flags_one,
)
import numpy as np
import pandas as pd
import UP_CIDDS_research_module.main.preprocessing_utils.B.decoding as B_decoder

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


def _decode_n_bits_cols(binary_cols: pd.DataFrame, binary_length: int) -> pd.Series:
    binaries = np.array([1 << n for n in range(binary_length - 1, -1, -1)])
    result = (binary_cols * binaries).sum(axis=1)
    return result


def _decode_B_WGAN_GP_ports(sixteen_cols):
    result = _decode_n_bits_cols(sixteen_cols, 16)
    return result.astype(int)


def decode_B_WGAN_GP(
    X, y, y_encoder, X_labels, X_encoders, is_decode_y=True
) -> pd.DataFrame:
    is_decode_y = True if y is not None else False
    full_dataset = np.hstack([X, y]) if is_decode_y else X
    if len(y_encoder.classes_) == 2:
        y_class_names = ["y"]
    else:
        y_class_names = [f"y_is_{c}" for c in y_encoder.classes_]
    full_df = pd.DataFrame(
        full_dataset, columns=(X_labels + y_class_names if is_decode_y else X_labels)
    )
    # decode duration
    full_df["Duration"] = X_encoders["Duration"].inverse_transform(
        full_df["Duration"].values.reshape(-1, 1)
    )

    # decode protocol
    proto_names = ["is_" + p for p in X_encoders["Protocols"].categories_[0]]
    full_df["Proto"] = X_encoders["Protocols"].inverse_transform(full_df[proto_names])
    full_df.drop(proto_names, axis=1, inplace=True)

    # decode date first seen
    full_df["day_seconds"] *= 86400
    time_to_str = full_df["day_seconds"].apply(
        lambda x: str(timedelta(seconds=round(x)))
    )

    days_names = X_encoders["Date_first_seen"].categories_[0]
    full_df["Date_first_seen"] = X_encoders["Date_first_seen"].inverse_transform(
        full_df[days_names]
    )

    full_df["Date_first_seen"] = (
        full_df["Date_first_seen"].apply(lambda x: x[3:]) + " " + time_to_str
    )  # x[3:] should be removeprefix("is_")
    full_df.drop(days_names, axis=1, inplace=True)
    full_df.drop("day_seconds", axis=1, inplace=True)

    # decode source IPs
    srcIP_names = [f"0bSrcIP{i}" for i in range(1, 32 + 1)]
    full_df["SrcIP"] = B_decoder._decode_B_WGAN_GP_ips(full_df[srcIP_names])
    full_df.drop(srcIP_names, axis=1, inplace=True)

    # decode dest IPs
    dstIP_names = [f"0bDstIP{i}" for i in range(1, 32 + 1)]
    full_df["DstIP"] = B_decoder._decode_B_WGAN_GP_ips(full_df[dstIP_names])
    full_df.drop(dstIP_names, axis=1, inplace=True)

    # decode SrcPt
    srcport_names = [f"0bSrcPt{i}" for i in range(1, 16 + 1)]
    full_df["SrcPt"] = B_decoder._decode_B_WGAN_GP_ports(full_df[srcport_names])
    full_df.drop(srcport_names, axis=1, inplace=True)

    # decode DstPt
    dstport_names = [f"0bDstPt{i}" for i in range(1, 16 + 1)]
    full_df["DstPt"] = B_decoder._decode_B_WGAN_GP_ports(full_df[dstport_names])
    full_df.drop(dstport_names, axis=1, inplace=True)

    # decode Bytes
    bytes_names = [f"0bBytes{i}" for i in range(1, 32 + 1)]
    full_df["Bytes"] = B_decoder._decode_n_bits_cols(full_df[bytes_names], 32).astype(
        int
    )
    full_df["Bytes"] = full_df["Bytes"].astype(int)
    full_df.drop(bytes_names, axis=1, inplace=True)

    # decode Packets
    packet_names = [f"0bPackets{i}" for i in range(1, 32 + 1)]
    full_df["Packets"] = B_decoder._decode_n_bits_cols(full_df[packet_names], 32)
    full_df["Packets"] = full_df["Packets"].astype(int)
    full_df.drop(packet_names, axis=1, inplace=True)

    # decode TCP flags
    flag_names = ["is_URG", "is_ACK", "is_PSH", "is_RES", "is_SYN", "is_FIN"]
    full_df["Flags"] = full_df[flag_names].apply(
        lambda x: _decode_TCP_flags_one(*x), axis=1
    )
    full_df.drop(flag_names, axis=1, inplace=True)

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
