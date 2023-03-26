from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler,
    LabelEncoder,
    LabelBinarizer,
)


def one_hot_encode_Proto(data: pd.Series) -> np.array:
    """one-hot encode the Proto column.
    Usage: onehotencoded_enc, onehotencoded_proto, proto_labels = one_hot_encode_Proto(subset["Proto"])


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
        enc,
        enc.transform(protocols).toarray(),
        list(enc.get_feature_names_out(["is"])),
    )


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
    """wrapper for min max scaler.
    Usage:
        scale: scaler, scaled = scale_min_max(data["Duration"])
        unscale: scaler.inverse_transform(scaled)
    Args:
        data (pd.DataFrame): a Series or multiple cols.

    Returns:
        np.array: Scaler.
        np.array: Scaled.
    """
    d = (
        data.to_numpy().reshape(-1, 1)
        if isinstance(data, pd.Series)
        else data.to_numpy()
    )
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(d)
    return scaler, scaled


def scale_ports(data: pd.DataFrame) -> np.array:
    d = (
        data.to_numpy().reshape(-1, 1)
        if isinstance(data, pd.Series)
        else data.to_numpy()
    )
    return d / 65535


def _preprocess_date_first_seen(dates_first_seen: pd.Series):
    """process date_first_seen attribute. It's the same for all preprocessing methods.
    Usage: _preprocess_date_first_seen(dataset.Date_first_seen)

    Args:
        dates_first_seen (pd.Series): the Date_first_seen column.
    """
    day_col_names = [
        "is_Monday",
        "is_Tuesday",
        "is_Wednesday",
        "is_Thursday",
        "is_Friday",
        "is_Saturday",
        "is_Sunday",
    ]
    dates = dates_first_seen.apply(
        lambda the_date: datetime.strptime(the_date, "%Y-%m-%d %H:%M:%S.%f")
    )
    days_of_week = (
        dates.apply(lambda the_date: day_col_names[the_date.weekday()])
        .to_numpy()
        .reshape(-1, 1)
    )
    seconds_of_day = (
        dates.apply(
            lambda t: t.hour * 60 * 60 + t.minute * 60 + t.second
        )  # ignore miliseconds
        .to_numpy()
        .reshape(-1, 1)
    ) / 86400
    enc = OneHotEncoder()
    enc.fit(days_of_week)
    return (
        enc,
        enc.transform(days_of_week).toarray(),
        list(enc.categories_[0]),
        seconds_of_day,
    )


def _random_IP_addr(random_seed=555, final_digit=0):
    # private IPs: 10.0.0.0 to 10.255.255.255
    # private IPs: 172.16.0.0 to 172.31.255.255
    # private IPs: 192.168.0.0 to 192.168.255.255
    # since the private IP addresses were not anonymised, we have to make sure we avoid accidentally generating private ones
    rng = np.random.default_rng(random_seed)

    first_num_choices = set(range(0, 256)) - set(
        (10,)
    )  # probably super inefficient lol
    first_num = rng.choice(list(first_num_choices))

    # determine second_num choices
    if first_num == 172:  # can't be [16,31], i.e. in [0,15] or [32,256]
        second_num_choices = set(range(0, 16)) | set(range(31 + 1, 256))
    elif first_num == 192:  # can't be 168
        second_num_choices = set(range(0, 256)) - set((168,))
    else:
        second_num_choices = set(range(0, 256))
    second_num = rng.choice(list(second_num_choices))
    third_num = rng.integers(0, 255 + 1)
    # fourth_num = rng.integers(0,255+1)
    return f"{first_num}.{second_num}.{third_num}.{final_digit}"


def _deanonymise_IP(ip: str) -> list:
    """convert anonymised IPs to ip strings.
    Ex: EXT_SERVER, OPENSTACK_NET, 39832_109 etc.

    Args:
        ip (str): the anonymised IP.

    Returns:
        list: the denonymised IP as its 4 bytes.
    """
    if ip == "EXT_SERVER":
        ip = f"555_127"
    elif ip == "OPENSTACK_NET":
        ip = f"777_187"
    elif ip == "DNS":
        ip = f"444_75"
    elif ip == "ATTACKER1":
        ip = f"111_64"
    elif ip == "ATTACKER2":
        ip = f"222_75"
    elif ip == "ATTACKER3":
        ip = f"333_2"
    front_back = ip.split("_")
    if len(front_back) == 1:
        return ip  # already an ip address
    front, back = int(front_back[0]), int(front_back[1])
    # use the front value as seed, so that unique values always map to the same randomly-generated IP.
    return _random_IP_addr(front, back)


# list of special anonymised IPs.
ANONYMISED_NAMED_IPS = {
    name: _deanonymise_IP(name)
    for name in [
        "EXT_SERVER",
        "OPENSTACK_NET",
        "DNS",
        "ATTACKER1",
        "ATTACKER2",
        "ATTACKER3",
    ]
}


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


def decode_TCP_flags_one(*six_flags):
    full_flags = "UAPRSF"
    flag = [full_flags[i] if six_flags[i] > 0.5 else "." for i in range(6)]
    return "".join(flag)


def decode_IP_N_WGAN_GP(four_cols):
    temp = (four_cols * 255).astype(int).astype(str)
    return temp.agg(lambda x: ".".join(x), axis=1)


def decode_N_WGAN_GP(X, y, y_encoder, labels, X_encoders):
    """Given X and y and their encoders, decode the data into traffic flow.
    If y is None, then only the X is decoded; in this case, y and y_encoder are ignored.

    Args:
        X (np.array): X.
        y (np.array | None): y. Set to None to ignore.
        y_encoder (_type_): _description_
        labels (list): the list of labels.
        X_encoders (_type_): _description_

    Returns:
        _type_: _description_
    """
    is_decode_y = True if y is not None else False
    full_dataset = np.hstack([X, y]) if is_decode_y else X
    y_class_names = [f"y_is_{c}" for c in y_encoder.classes_]
    full_df = pd.DataFrame(
        full_dataset, columns=(labels + y_class_names if is_decode_y else labels)
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
    ports_normed = scale_ports(subset[["SrcPt", "DstPt"]])
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

    if include_date_ip:  # when including date and ip, fetch their columns as well
        src_ips = _process_N_WGAN_GP_ips(subset["SrcIP"])
        dest_ips = _process_N_WGAN_GP_ips(subset["DstIP"])
        (
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

    return full_X, y, y_encoder, labels
