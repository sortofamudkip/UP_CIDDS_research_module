import sys, pytest
import numpy as np
import pandas as pd

# sys.path.append("../main/preprocessing_utils/general")  # Adds higher directory to python modules path.

from ..main.preprocessing_utils.general.encode import (
    one_hot_encode_Proto,
    one_hot_encode_TCP_Flags,
)

from ..main.preprocessing_utils.N.encode import scale_ports_N


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
    _, onehotencoded_proto, _ = one_hot_encode_Proto(protocols)
    assert (onehotencoded_proto == expected).all()


def test_one_hot_encode_TCP_Flags():
    flags = pd.Series(["......", ".AP.S.", "UAPRSF", "...R.."])
    expected = np.array(
        [[0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0]]
    )
    onehotencoded_tcpflags, _ = one_hot_encode_TCP_Flags(flags)
    assert (onehotencoded_tcpflags == expected).all()


def test_scale_ports_N():
    ports = pd.Series(range(0, 65535 + 1))
    expected_output = (ports / 65535).to_numpy().reshape(-1, 1)
    assert scale_ports_N(ports) == pytest.approx(expected_output)
