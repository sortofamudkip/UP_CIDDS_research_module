import sys, pytest

sys.path.append("../main/")  # Adds higher directory to python modules path.

import load_data


def test_load_data_len():
    data = load_data.load_data()
    assert len(data) == 671241


@pytest.mark.parametrize(
    "hex_str, TCP_string_result",
    [
        ("0xde", ".APRS."),
        ("0xd2", ".A..S."),
        ("0x0", "......"),
        ("0x3f", "UAPRSF"),
        ("0xff", "UAPRSF"),
    ],
)
def test_hex_string_to_TCP_flags(hex_str, TCP_string_result):
    assert load_data._hex_string_to_TCP_flags(hex_str) == TCP_string_result
