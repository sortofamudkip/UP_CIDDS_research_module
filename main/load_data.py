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


def testt():
    return EXTERNAL_DATASET_DIR_NAME


def load_data_raw() -> pd.DataFrame:
    """Loads data with no preprocessing.

    Returns:
        pd.DataFrame: the dataset.
    """
    dataset = pd.concat(
        [
            pd.read_csv(
                f"{EXTERNAL_DATASET_DIR_NAME}/CIDDS-001-external-week{i}.csv",
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
    )
    return dataset


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drops unnecessary columns from dataset.

    Args:
        df (pd.DataFrame): original dataset.

    Returns:
        pd.DataFrame: new dataset.
    """
    return df.drop(
        ["Flows", "Tos", "attackType", "attackID", "attackDescription"], axis=1
    ).drop(
        ["SrcIP", "DstIP", "Date_first_seen"], axis=1
    )  # dropped due to anonymisation and not being useful


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


def clean_data(dataset):
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


def load_data():
    return drop_unnecessary_columns(clean_data(load_data_raw()))


if __name__ == "__main__":
    data = load_data()
    print(len(data))
