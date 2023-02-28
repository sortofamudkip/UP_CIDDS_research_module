import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualise_PCA(
    X, y, title="Projected dataset using sklearn's PCA(2)", savefig=False
) -> None:
    """Given X and y,
    usage:
        X, y = preprocessor.get_N_WGAN_GP_preprocessed_data(data, binary_labels=True)
        visualise_PCA(X, y)

    Args:
        X (np.array): Attributes.
        y (np.array): Classes.
        title (str): Title of graph.
        savefig (boolean): whether to save the graph or not.

    """
    y = y.ravel()
    pca = PCA(n_components=2)
    pca.fit(X)

    # obtain projection
    projection = X @ pca.components_.T

    # plot
    projection_with_labels = pd.DataFrame(projection, columns=["x", "y"])
    projection_with_labels["class"] = y
    for label in projection_with_labels["class"].unique():
        subset = projection_with_labels[projection_with_labels["class"] == label]
        plt.scatter(x=subset.x, y=subset.y, label=label, s=1)
    plt.title(title)
    plt.legend()
    if savefig:
        plt.savefig()
