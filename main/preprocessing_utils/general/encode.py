from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import pandas as pd
from datetime import datetime


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
    Usage: onehotencoded_tcpflags, labels_tcpflags = one_hot_encode_TCP_Flags(data["Flags"])

    Args:
        data (pd.Series): _description_

    Returns:
        np.array: np.array of shape (num_rows, 4).
    """
    onehotencoded = (
        data.str.split("", expand=True)
        .loc[:, 1:6]
        .applymap(lambda x: 0 if x == "." else 1)
        .to_numpy()
    )
    labels = ["is_URG", "is_ACK", "is_PSH", "is_RES", "is_SYN", "is_FIN"]
    return onehotencoded, labels


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
