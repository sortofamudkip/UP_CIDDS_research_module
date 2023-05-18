import pandas as pd
import re
import numpy as np
from tensorflow import keras

# example usage:
#   data = pd.read_csv('external_denonymisedIPs.csv.zip', compression='zip', index_col=0)


############################################################
#                                                          #
#    These tests pass (score 100%) on external dataset     #
#                                                          #
############################################################


# "Test  (Ring et al 2018)"
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
    Packets = subset.Packets.astype(np.int64)
    Bytes = subset.Bytes
    condition = (
        ((Packets > 0) & (Bytes > 0))
        & ((20 * Packets) <= Bytes)
        & (Bytes <= (65535 * Packets))
    )
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


def score_diversity(data: pd.DataFrame) -> float:
    """Tests how diverse the y labels are.
    If only one class is present, this test scores 0.
    If all five classes are present, this test scores 1.0.
    In other words, the score is 0.25 * (number of unique labels - 1).

    Args:
        data (pd.DataFrame): the synthetic dataset.

    Returns:
        float: the score.
    """
    if not all(col in data.columns for col in ("class",)):
        return None  # skip test
    return 0.25 * (len(data["class"].unique()) - 1)


############################################################
#                                                          #
#   These tests fail (score <100%) on external dataset     #
#                                                          #
############################################################


# "Test 1 (Ring et al 2018)"
def score_if_udp_no_tcp_flags(data: pd.DataFrame) -> float:
    if not all(col in data.columns for col in ("Proto", "class", "Flags")):
        return None  # skip test
    subset = data[(data["Proto"] == "UDP") & (data["class"] == "normal")]
    if len(subset) == 0:
        return False
    condition = subset["Flags"] == "......"
    return condition.sum() / len(subset)


# "Test 4 (Ring et al 2018)"
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
#             Auxilliary functions / constants             #
#                                                          #
############################################################


def _validate_ipv4(ip: str) -> bool:
    m = re.match(r"(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$", ip)
    if not m:
        return False
    if len(m.groups()) != 4:
        return False
    return all(0 <= int(quartet) <= 255 for quartet in ip.split("."))


LIST_OF_REALISTIC_DATASET_TESTS = {
    "score_normal_http_is_tcp": score_normal_http_is_tcp,
    "score_packet_size": score_packet_size,
    "score_IPs_in_range": score_IPs_in_range,
    "score_numerics_valid": score_numerics_valid,
    "score_ports_valid": score_ports_valid,
    "score_diversity": score_diversity,
}


def score_data_plausibility_detailed(data):
    report = {
        scorename: [scorefunc(data)]
        for scorename, scorefunc in LIST_OF_REALISTIC_DATASET_TESTS.items()
    }
    return pd.DataFrame(report)


def score_data_plausibility_single(data, verbose=True):
    report = [scorefunc(data) for scorefunc in LIST_OF_REALISTIC_DATASET_TESTS.values()]
    if verbose:
        print(
            f"\nreport:\n\t{[(name, r) for r, name in zip(report, LIST_OF_REALISTIC_DATASET_TESTS.keys())]}"
        )
    report = [score for score in report if not (score is None or score is False)]
    return np.nanmean(np.array(report))


class EvaluateSyntheticDataRealisticnessCallback(keras.callbacks.Callback):
    """
    A Keras callback that generates a fake dataset using the Generator
    and then evaluates how plausible to would be based on the score_data_realness_single() score.
    This callback only runs at the end of each epoch.
    Usage:
    self.gan.fit(self.dataset, epochs=epochs, callbacks=[EvaluateSyntheticDataRealisticnessCallback(model, x_test, y_test)])
    """

    def __init__(
        self, model, generate_samples_func, num_samples_to_generate, decoder_func
    ):
        self.model = model
        self.generate_samples_func = generate_samples_func
        self.num_samples_to_generate = num_samples_to_generate
        self.decoder_func = decoder_func

    def on_epoch_end(self, epoch, logs={}):
        # print(f"epoch {epoch}, logs {logs}, model {self.model}")
        generated_samples = self.generate_samples_func(self.num_samples_to_generate)
        decoded = self.decoder_func(generated_samples)
        score = score_data_plausibility_single(decoded)
        print(f"\trealistic dataset score: {score}")
