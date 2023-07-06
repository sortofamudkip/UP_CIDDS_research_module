import pandas as pd
import re
import numpy as np
from tensorflow import keras
import pickle
from pathlib import Path

# example usage:
#   data = pd.read_csv('external_denonymisedIPs.csv.zip', compression='zip', index_col=0)


############################################################
#                                                          #
#    These tests pass (score 100%) on external dataset     #
#                                                          #
############################################################


# "Test 1 (Ring et al 2018)"
# note: we don't need to check label because UDP physically cannot have TCP flags,
#       due to the nature of UDP headers.
#       See: https://en.wikipedia.org/wiki/User_Datagram_Protocol
def score_if_udp_no_tcp_flags(data: pd.DataFrame, num_classes: int, return_mask=False):
    if not all(col in data.columns for col in ("Proto", "Flags")):
        return None  # skip test
    prerequisite = data["Proto"] == "UDP"
    if return_mask:
        mask = np.ones(len(data), dtype=bool)  # all true
        mask[prerequisite & (data["Flags"] != "......")] = False
        return mask
    else:
        subset = data[prerequisite]
        if len(subset) == 0:
            return False
        condition = subset["Flags"] == "......"
        return condition.sum() / len(subset)


# Test 2 is not implemented because it's not applicable to the external dataset


# "Test 3 (Ring et al 2018)"
def score_normal_http_is_tcp(data: pd.DataFrame, num_classes: int, return_mask=False):
    if not all(col in data.columns for col in ("SrcPt", "class", "DstPt")):
        return None  # skip test
    prerequisite = (
        (data.SrcPt == 80)  # HTTP
        | (data.SrcPt == 443)  # HTTPS
        | (data.DstPt == 80)  # HTTP
        | (data.DstPt == 443)  # HTTPS
    ) & (data["class"] == "normal")
    if return_mask:
        mask = np.ones(len(data), dtype=bool)  # all true
        mask[prerequisite & (data["Proto"] != "TCP")] = False
        return mask
    else:
        subset = data[prerequisite]
        if len(subset) == 0:
            return False
        condition = subset["Proto"] == "TCP"
        return condition.sum() / len(subset)


# Test 4 returns False on the external dataset (see below)

# Test 5 is not implemented because it's not applicable to the external dataset

# Test 6 is not implemented because it's not applicable to the external dataset


# "Test 7 (Ring et al 2018)"
# note that the test in the paper is not completely correct.
# the correct limits are
def score_packet_size(data: pd.DataFrame, num_classes: int, return_mask=False):
    if not all(col in data.columns for col in ("Proto", "Packets", "Bytes")):
        return None  # skip test
    prerequisite = (
        (data.Proto == "TCP") | (data.Proto == "UDP") | (data.Proto == "ICMP")
    )
    if return_mask:
        Packets = data["Packets"]
        Bytes = data["Bytes"]
        mask = np.ones(len(data), dtype=bool)  # all true
        mask[
            prerequisite
            & ~(
                (((Packets > 0) & (Bytes > 0)))
                & (
                    (
                        (data.Proto == "TCP")
                        & ((40 * Packets) <= Bytes)
                        & (Bytes <= (65535 * Packets))
                    )
                    | (
                        (data.Proto == "UDP")
                        & ((8 * Packets) <= Bytes)
                        & (Bytes <= (65535 * Packets))
                    )
                    | (
                        (data.Proto == "ICMP")
                        & ((64 * Packets) <= Bytes)
                        & (Bytes <= (65535 * Packets))
                    )
                )
            )
        ] = False
        return mask

    else:
        subset = data[prerequisite]
        if len(subset) == 0:
            return False
        Packets = subset.Packets.astype(
            np.int64
        )  # since 65535 * might overflow if int32
        Bytes = subset.Bytes
        condition_tcp = (
            (subset.Proto == "TCP")
            & ((Packets > 0) & (Bytes > 0))
            & ((40 * Packets) <= Bytes)
            & (Bytes <= (65535 * Packets))
        )
        condition_udp = (
            (subset.Proto == "UDP")
            & ((Packets > 0) & (Bytes > 0))
            & ((8 * Packets) <= Bytes)
            & (Bytes <= (65535 * Packets))
        )
        condition_icmp = (
            (subset.Proto == "ICMP")
            & ((Packets > 0) & (Bytes > 0))
            & ((64 * Packets) <= Bytes)
            & (Bytes <= (65535 * Packets))
        )
        condition = condition_tcp | condition_udp | condition_icmp
        return condition.sum() / len(subset)


############################################################
#                                                          #
#        Self-written tests (pass external dataset)        #
#                                                          #
############################################################


def score_IPs_in_range(data: pd.DataFrame, num_classes: int, return_mask=False):
    if not all(col in data.columns for col in ("SrcIP", "DstIP")):
        return None  # skip test
    srcIPs_valid = data["SrcIP"].apply(_validate_ipv4)
    dstIPs_valid = data["DstIP"].apply(_validate_ipv4)
    condition = srcIPs_valid & dstIPs_valid
    return condition.sum() / len(data) if not return_mask else condition


def score_numerics_valid(data: pd.DataFrame, num_classes: int, return_mask=False):
    if not all(col in data.columns for col in ("Duration", "Packets", "Bytes")):
        return None  # skip test
    subset = data[["Duration", "Packets", "Bytes"]]
    if len(subset) == 0:
        return False
    condition = (data.Duration >= 0) & (data.Packets > 0) & (data.Bytes > 0)
    return condition.sum() / len(subset) if not return_mask else condition


def score_ports_valid(data: pd.DataFrame, num_classes: int, return_mask=False):
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
    return condition.sum() / len(subset) if not return_mask else condition


def score_diversity(data: pd.DataFrame, num_classes: int, return_mask=False):
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
    factor = 0.25 if num_classes == 5 else 1
    return factor * (len(data["class"].unique()) - 1)


############################################################
#                                                          #
#   These tests fail (score <100%) on external dataset     #
#                                                          #
############################################################


# "Test 4 (Ring et al 2018)"
# This test fails because subset is empty, i.e. it returns False.
def score_normal_dns_is_UDP(data: pd.DataFrame, num_classes: int, return_mask=False):
    if not all(col in data.columns for col in ("SrcPt", "class", "DstPt")):
        return None  # skip test
    prerequisite = ((data.SrcPt == 53) | (data.DstPt == 53)) & (
        data["class"] == "normal"
    )
    if return_mask:
        mask = np.ones(len(data), dtype=bool)  # all true
        mask[prerequisite & (data["Proto"] != "UDP")] = False
        return mask
    else:
        subset = data[prerequisite]
        if len(subset) == 0:
            return False
        condition = subset["Proto"] == "UDP"
        return condition.sum() / len(subset)


############################################################
#                                                          #
#              Scoring functions / constants               #
#                                                          #
############################################################

# score_if_udp_no_tcp_flags is turned off since already postprocessed
LIST_OF_REALISTIC_DATASET_TESTS = {
    # "score_if_udp_no_tcp_flags": score_if_udp_no_tcp_flags,
    "score_normal_http_is_tcp": score_normal_http_is_tcp,
    "score_packet_size": score_packet_size,
    "score_IPs_in_range": score_IPs_in_range,
    "score_numerics_valid": score_numerics_valid,
    "score_ports_valid": score_ports_valid,
    "score_diversity": score_diversity,
}


def score_data_plausibility_detailed(data, num_classes: int):
    report = {
        scorename: [scorefunc(data, num_classes)]
        for scorename, scorefunc in LIST_OF_REALISTIC_DATASET_TESTS.items()
    }
    return pd.DataFrame(report).rename({0: "score"})


def score_data_plausibility_single(data, num_classes: int, verbose=True):
    report = [
        scorefunc(data, num_classes)
        for scorefunc in LIST_OF_REALISTIC_DATASET_TESTS.values()
    ]
    report_dict = dict(
        [(name, r) for r, name in zip(report, LIST_OF_REALISTIC_DATASET_TESTS.keys())]
    )
    if verbose:
        print(f"report: {report_dict}")
    if "score_diversity" in report_dict and report_dict["score_diversity"] == 0:
        print("\t⚠️ Warning: only one unique label!")
    report = [score for score in report if not (score is None or score is False)]
    return np.nanmean(np.array(report))


def mask_plausible_rows(data, num_classes: int, verbose=True):
    # note: num_classes is ignored since score_diversity is not tested
    bool_cols = pd.concat(
        [
            pd.Series(scorefunc(data, num_classes, return_mask=True), dtype=float)
            for scorefunc in LIST_OF_REALISTIC_DATASET_TESTS.values()
            if scorefunc != score_diversity
        ],
        axis=1,
    )
    return bool_cols.all(axis=1)
    # return [
    #     scorefunc(data, num_classes, return_mask=True)
    #     for scorefunc in LIST_OF_REALISTIC_DATASET_TESTS.values()
    #     if scorefunc != score_diversity
    # ],


def filter_plausible_rows(data, num_classes: int, verbose=True):
    mask = mask_plausible_rows(data, num_classes, verbose)
    valid_rows = data[mask]
    retention_ratio = len(valid_rows) / len(data)
    return valid_rows, retention_ratio


class EvaluateSyntheticDataRealisticnessCallback(keras.callbacks.Callback):
    """
    A Keras callback that generates a fake dataset using the Generator
    and then evaluates how plausible to would be based on the score_data_realness_single() score.
    This callback only runs at the end of each epoch.
    Usage:
    self.gan.fit(self.dataset, epochs=epochs, callbacks=[EvaluateSyntheticDataRealisticnessCallback(model, x_test, y_test)])
    """

    def __init__(
        self,
        model,
        generate_samples_func,
        num_samples_to_generate,
        decoder_func,
        pipeline_name,
    ):
        self.model = model
        self.generate_samples_func = generate_samples_func
        self.num_samples_to_generate = num_samples_to_generate
        self.decoder_func = decoder_func
        self.pipeline_name = pipeline_name

    def on_epoch_end(self, epoch, logs={}):
        # print(f"epoch {epoch}, logs {logs}, model {self.model}")
        generated_samples = self.generate_samples_func(self.num_samples_to_generate)
        # decoded = self.decoder_func(generated_samples)
        # score = score_data_plausibility_single(decoded)
        # save dataset for future evaluation
        output_path = (
            Path(__file__).parent
            / f"../../results/"
            / self.pipeline_name
            / f"synthetic_epoch{epoch+1}.npy"
        )
        np.save(output_path, generated_samples)
        # print(f"\trealistic dataset score: {score}")


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
