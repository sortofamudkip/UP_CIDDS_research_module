from pandas import DataFrame


def postprocess_UDP_TCP_flags(dataset: DataFrame) -> DataFrame:
    """Given a dataset, set the TCP flags of all UDP rows to "......" (no TCP flags).
    This only applies to datasets that are already decoded from numpy!
    This function is NOT USED during GAN training.

    Args:
        dataset (DataFrame): the human-readable dataset (a pandas df).

    Returns:
        DataFrame: the post-processed pandas df.
    """
    dataset.loc[dataset["Proto"] == "UDP", "Flags"] = "......"
    return dataset
