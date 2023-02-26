import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

EXTERNAL_DATASET_DIR_NAME = os.getenv("EXTERNAL_DATASET_DIR_NAME")


def load_data_raw() -> pd.DataFrame:
    """Loads data with no preprocessing.

    Returns:
        pd.DataFrame: the dataset.
    """
    dataset = pd.concat(
        [
            pd.read_csv(
                f"{EXTERNAL_DATASET_DIR_NAME}CIDDS-001-external-week{i}.csv",
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
    )


def load_data():
    return drop_unnecessary_columns(load_data_raw())


if __name__ == "__main__":
    data = load_data()
    print(len(data))
