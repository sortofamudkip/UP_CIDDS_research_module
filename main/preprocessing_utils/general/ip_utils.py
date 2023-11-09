import numpy as np
import pandas as pd


class IP_Preprocessor:
    def __init__(self) -> None:
        self.ip_dict = {}

    def fit(self, ip_col: pd.Series) -> None:
        """Fit the preprocessor to the given IP column.

        Args:
            ip_col (pd.Series): the IP column to fit to.
        """
        unique_ips_in_col = ip_col.unique()
        self.ip_dict = self._get_ip_dict(unique_ips_in_col)

    def _get_ip_dict(self, ip_col: pd.Series) -> dict:
        """Get the IP dictionary from the given IP column.

        Args:
            ip_col (pd.Series): the IP column to get the dictionary from.

        Returns:
            dict: the IP dictionary.
        """
        ip_dict = {}
        for ip in ip_col:
            if ip not in ip_dict:
                ip_dict[ip] = _deanonymise_IP(ip)
        return ip_dict

    def transform(self, ip_col: pd.Series) -> pd.Series:
        """Transform the given IP column.

        Args:
            ip_col (pd.Series): the IP column to transform.

        Returns:
            pd.Series: the transformed IP column.
        """
        return ip_col.apply(self._transform_ip)

    def _transform_ip(self, ip: str) -> str:
        """Transform the given IP.

        Args:
            ip (str): the IP to transform.

        Returns:
            str: the transformed IP.
        """
        return self.ip_dict[ip]

    def fit_transform(self, ip_col: pd.Series) -> pd.Series:
        """Fit and transform the given IP column.

        Args:
            ip_col (pd.Series): the IP column to fit and transform.

        Returns:
            pd.Series: the transformed IP column.
        """
        self.fit(ip_col)
        return self.transform(ip_col)


def _random_IP_addr(random_seed=555, final_digit=0):
    # private IPs: 10.0.0.0 to 10.255.255.255
    # private IPs: 172.16.0.0 to 172.31.255.255
    # private IPs: 192.168.0.0 to 192.168.255.255
    # since the private IP addresses were not anonymised, we have to make sure we avoid accidentally generating private ones
    rng = np.random.default_rng(random_seed)

    first_num_choices = set(range(0, 256)) - set(
        (10,)
    )  # probably super inefficient lol
    first_num = rng.choice(list(first_num_choices))

    # determine second_num choices
    if first_num == 172:  # can't be [16,31], i.e. in [0,15] or [32,256]
        second_num_choices = set(range(0, 16)) | set(range(31 + 1, 256))
    elif first_num == 192:  # can't be 168
        second_num_choices = set(range(0, 256)) - set((168,))
    else:
        second_num_choices = set(range(0, 256))
    second_num = rng.choice(list(second_num_choices))
    third_num = rng.integers(0, 255 + 1)
    # fourth_num = rng.integers(0,255+1)
    return f"{first_num}.{second_num}.{third_num}.{final_digit}"


def _deanonymise_IP(ip: str) -> list:
    """convert anonymised IPs to ip strings.
    Ex: EXT_SERVER, OPENSTACK_NET, 39832_109 etc.

    Args:
        ip (str): the anonymised IP.

    Returns:
        list: the denonymised IP as its 4 bytes.
    """
    if ip == "EXT_SERVER":
        ip = f"555_127"
    elif ip == "OPENSTACK_NET":
        ip = f"777_187"
    elif ip == "DNS":
        ip = f"444_75"
    elif ip == "ATTACKER1":
        ip = f"111_64"
    elif ip == "ATTACKER2":
        ip = f"222_75"
    elif ip == "ATTACKER3":
        ip = f"333_2"
    front_back = ip.split("_")
    if len(front_back) == 1:
        return ip  # already an ip address
    front, back = int(front_back[0]), int(front_back[1])
    # use the front value as seed, so that unique values always map to the same randomly-generated IP.
    return _random_IP_addr(front, back)


# list of special anonymised IPs.
ANONYMISED_NAMED_IPS = {
    name: _deanonymise_IP(name)
    for name in [
        "EXT_SERVER",
        "OPENSTACK_NET",
        "DNS",
        "ATTACKER1",
        "ATTACKER2",
        "ATTACKER3",
    ]
}
