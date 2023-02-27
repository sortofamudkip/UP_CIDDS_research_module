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
    flags = pd.Series(["......", ".AP.S.", "UAPRSF", "...R.."])
    expected = np.array(
        [[0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0]]
    )
    assert (preprocess_data.one_hot_encode_TCP_Flags(flags) == expected).all()


def test_scale_ports_1d():
    ports = pd.Series(range(0, 65535 + 1))
    expected_output = (ports / 65535).to_numpy().reshape(-1, 1)
    assert preprocess_data.scale_ports(ports) == pytest.approx(expected_output)
