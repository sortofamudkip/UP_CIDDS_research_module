import pandas as pd
import re

# example usage:
#   data = pd.read_csv('external_denonymisedIPs.csv.zip', compression='zip', index_col=0)

############################################################
#                                                          #
#    These tests pass (score 100%) on external dataset     #
#                                                          #
############################################################


def score_normal_http_is_tcp(data: pd.DataFrame) -> float:
    if not all(col in data.columns for col in ("SrcPt", "class", "DstPt")):
        return None  # skip test
    subset = data[
        (
            (data.SrcPt == 80)
            | (data.SrcPt == 443)
            | (data.DstPt == 80)
            | (data.DstPt == 443)
        )
        & (data["class"] == "normal")
    ]
    if len(subset) == 0:
        return False
    condition = subset["Proto"] == "TCP"
    return condition.sum() / len(subset)


def score_packet_size(data: pd.DataFrame) -> float:
    if not all(col in data.columns for col in ("Proto", "Packets", "Bytes")):
        return None  # skip test
    subset = data[
        ((data.Proto == "TCP") | (data.Proto == "UDP") | (data.Proto == "ICMP"))
    ]
    if len(subset) == 0:
        return False
    packets = subset.Packets.astype(np.int64)
    condition = ((20 * packets) <= subset.Bytes) & (subset.Bytes <= (65535 * packets))
    # return subset[ ~condition ][ ["Packets", "Bytes", "class"] ]
    return condition.sum() / len(subset)


def score_IPs_in_range(data: pd.DataFrame) -> float:
    if not all(col in data.columns for col in ("SrcIP", "DstIP")):
        return None  # skip test
    srcIPs_valid = data["SrcIP"].apply(_validate_ipv4)
    dstIPs_valid = data["DstIP"].apply(_validate_ipv4)
    condition = srcIPs_valid & dstIPs_valid
    return condition.sum() / len(data)


############################################################
#                                                          #
#        Self-written tests (pass external dataset)        #
#                                                          #
############################################################


def score_numerics_valid(data: pd.DataFrame) -> float:
    if not all(col in data.columns for col in ("Duration", "Packets", "Bytes")):
        return None  # skip test
    subset = data[["Duration", "Packets", "Bytes"]]
    if len(subset) == 0:
        return False
    condition = (data.Duration >= 0) & (data.Packets > 0) & (data.Bytes > 0)
    return condition.sum() / len(subset)


def score_ports_valid(data: pd.DataFrame) -> float:
    if not all(col in data.columns for col in ("SrcPt", "DstPt")):
        return None  # skip test
    subset = data[["SrcPt", "DstPt"]]
    if len(subset) == 0:
        return False
    condition = (
        (0 <= subset.SrcPt)
        & (subset.SrcPt <= 65535)
        & (0 <= subset.DstPt)
        & (subset.DstPt <= 65535)
    )
    return condition.sum() / len(subset)


############################################################
#                                                          #
#   These tests fail (score <100%) on external dataset     #
#                                                          #
############################################################


def score_if_udp_no_tcp_flags(data: pd.DataFrame) -> float:
    if not all(col in data.columns for col in ("Proto", "class", "Flags")):
        return None  # skip test
    subset = data[(data["Proto"] == "UDP") & (data["class"] == "normal")]
    if len(subset) == 0:
        return False
    condition = subset["Flags"] == "......"
    return condition.sum() / len(subset)


def score_normal_dns_is_UDP(data: pd.DataFrame) -> float:
    if not all(col in data.columns for col in ("SrcPt", "class", "DstPt")):
        return None  # skip test
    subset = data[
        ((data.SrcPt == 53) | (data.DstPt == 53)) & (data["class"] == "normal")
    ]
    if len(subset) == 0:
        return False
    condition = subset["Proto"] == "UDP"
    return condition.sum() / len(subset)


############################################################
#                                                          #
#                   Auxilliary functions                   #
#                                                          #
############################################################


def _validate_ipv4(ip: str) -> bool:
    m = re.match(r"(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$", ip)
    if not m:
        return False
    if len(m.groups()) != 4:
        return False
    return all(0 <= int(quartet) <= 255 for quartet in ip.split("."))
