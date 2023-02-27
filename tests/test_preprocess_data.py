import sys, pytest
import numpy as np
import pandas as pd

sys.path.append("../main/")  # Adds higher directory to python modules path.

import preprocess_data


def test_one_hot_encode_Proto():
    protocols = pd.Series(["TCP", "UDP", "ICMP", "GRE", "TCP"])
    expected = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    assert (preprocess_data.one_hot_encode_Proto(protocols) == expected).all()


def test_one_hot_encode_TCP_Flags():
    flags = pd.Series(["......", ".AP.S.", "UAPRSF"])
    expected = np.array([[0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1]])
    assert (preprocess_data.one_hot_encode_TCP_Flags(flags) == expected).all()
