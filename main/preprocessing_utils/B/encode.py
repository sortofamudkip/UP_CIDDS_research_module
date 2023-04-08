import numpy as np
import pandas as pd


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
