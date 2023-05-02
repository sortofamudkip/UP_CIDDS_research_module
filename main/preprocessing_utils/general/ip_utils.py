import numpy as np


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
