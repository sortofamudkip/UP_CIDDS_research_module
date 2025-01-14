import re
import pandas as pd
import os
from dotenv import load_dotenv

# Get the path to the directory this file is in
BASEDIR = os.path.abspath(os.path.dirname(__file__))

# Connect the path with your '.env' file name
load_dotenv(os.path.join(BASEDIR, ".env"))

EXTERNAL_DATASET_DIR_NAME = os.path.abspath(
    os.path.join(BASEDIR, "..", os.getenv("EXTERNAL_DATASET_DIR_NAME"))
)
INTERNAL_DATASET_DIR_NAME = os.path.abspath(
    os.path.join(BASEDIR, "..", os.getenv("INTERNAL_DATASET_DIR_NAME"))
)


def load_data_raw(dataset: str = "external") -> pd.DataFrame:
    """Loads data with no preprocessing.

    Returns:
        pd.DataFrame: the dataset.
    """
    assert dataset in ("external", "internal")
    dataset_path = (
        EXTERNAL_DATASET_DIR_NAME
        if dataset == "external"
        else INTERNAL_DATASET_DIR_NAME
    )
    dataset = pd.concat(
        [
            pd.read_csv(
                f"{dataset_path}/CIDDS-001-{dataset}-week{i}.csv",
                dtype={
                    "Date first seen": str,
                    "Duration": float,
                    "Proto": "string",
                    "Src IP Addr": "string",
                    "Src Pt": int,
                    "Dst IP Addr": "string",
                    "Dst Pt": float,
                    "Packets": int,
                    "Bytes": "string",
                    "Flows": int,
                    "Flags": "string",
                    "Tos": int,
                    "class": "string",
                    "attackType": "string",
                    "attackID": "string",
                    "attackDescription": "string",
                },
            ).rename(
                columns={
                    "Date first seen": "Date_first_seen",
                    "Src IP Addr": "SrcIP",
                    "Src Pt": "SrcPt",
                    "Dst IP Addr": "DstIP",
                    "Dst Pt": "DstPt",
                }
            )
            for i in range(1, 4 + 1)
        ]
    ).reset_index(drop=True)
    return dataset


def drop_unnecessary_columns(
    df: pd.DataFrame,
    drop_date_IP: bool,
    is_external: bool = True,
) -> pd.DataFrame:
    """Drops unnecessary columns from dataset.

    Args:
        df (pd.DataFrame): original dataset.
        drop_date_IP (bool): drop SrcIP, DstIP, Date_first_seen

    Returns:
        pd.DataFrame: new dataset.
    """
    columns_to_drop = (
        ["Flows", "Tos", "attackType", "attackID", "attackDescription"]
        if is_external
        else ["Flows", "Tos", "attackID", "attackDescription"]
    )
    data = df.drop(columns_to_drop, axis=1)
    # dropped due to anonymisation and not being useful
    return (
        data.drop(["SrcIP", "DstIP", "Date_first_seen"], axis=1)
        if drop_date_IP
        else data
    )


def _hex_string_to_TCP_flags(hex_str: str):
    FULL_FLAGS = "UAPRSF"
    binary_representation = f"{int(hex_str, 16):0>8b}"[2:]
    TCP_representation = [
        FULL_FLAGS[i] if binary_representation[i] == "1" else "." for i in range(6)
    ]
    return "".join(TCP_representation)


def _bytes_string_to_int(bytes_str: str) -> int:
    if re.match(r"[0-9]+$", bytes_str):
        return int(bytes_str)
    value, unit = bytes_str.split(" ")
    assert unit == "M"  # assuming this is the only one atm
    return int(float(value) * 10**6)


def clean_data(dataset: pd.DataFrame):
    # strip whitespace from protocol number and Bytes
    dataset.Proto = dataset.Proto.apply(lambda x: x.strip())
    dataset["Bytes"] = dataset["Bytes"].apply(lambda x: x.strip())
    dataset["Flags"] = dataset["Flags"].apply(lambda x: x.strip())

    # force DstPt to be int (ICMP's DstPt is float for some reason)
    dataset["DstPt"] = dataset["DstPt"].astype(int)

    # process the unprocessed flags
    dataset["Flags"] = dataset["Flags"].apply(
        lambda flag: _hex_string_to_TCP_flags(flag) if flag.startswith("0x") else flag
    )

    # turn the number of bytes (string) into int
    dataset["Bytes"] = dataset["Bytes"].apply(_bytes_string_to_int)
    return dataset


def load_data(drop_date_IP, drop_misc_protocols, dataset: str = "external"):
    raw_data = load_data_raw(dataset)
    data = drop_unnecessary_columns(
        clean_data(raw_data), drop_date_IP, dataset == "external"
    )
    return (
        data[data["Proto"].isin(["ICMP", "TCP", "UDP"])]
        if drop_misc_protocols
        else data
    )


if __name__ == "__main__":
    data = load_data()
    print(len(data))
