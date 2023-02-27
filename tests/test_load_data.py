import sys

sys.path.append("../main/")  # Adds higher directory to python modules path.

import load_data


def test_load_data_len():
    data = load_data.load_data()
    assert len(data) == 671241
