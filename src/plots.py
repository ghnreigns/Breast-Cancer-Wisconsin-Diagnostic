import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
from typing import List
import numpy as np
from sklearn import decomposition, manifold, preprocessing


def plot_target_distribution(df: pd.DataFrame, target: str, colors: List[str]) -> None:
    """Plot Target Distribution with percentage labels.

    Args:
        df (pd.DataFrame): [description]
        target (str): [description]
        colors (List[str]): [description]
    """

    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300

    x_axis = df["diagnosis"].value_counts().index
    y_axis = df["diagnosis"].value_counts()

    figure, target_bar = plt.subplots(figsize=(6, 4))
    sns.barplot(x=x_axis, y=y_axis, ax=target_bar, palette={1: colors[0], 0: colors[3]})
    target_bar.set_xticklabels(["Benign", "Malignant"])
    target_bar.set_ylabel("Frequency Count")
    target_bar.legend(["Benign", "Malignant"], loc="upper right")
    target_bar.set_title("Count of Target (Diagnosis)", fontsize=16)
    figure.text(
        x=0.27,
        y=0.8,
        s="{:.1f}%".format(df["diagnosis"].value_counts(normalize=True)[0] * 100),
        **{"weight": "bold", "color": "black"},
    )
    figure.text(
        x=0.66,
        y=0.5,
        s="{:.1f}%".format(df["diagnosis"].value_counts(normalize=True)[1] * 100),
        **{"weight": "bold", "color": "black"},
    )
    plt.show()


def plot_univariate(df: pd.DataFrame, predictor: str, colors: List[str]) -> None:
    """Take in continuous predictors and plot univariate distribution. Note in this setting, we have kde=True.

    Args:
        df (pd.DataFrame): [description]
        predictor (str): [description]
    """

    univariate_params = {"nrows": 10, "ncols": 3, "figsize": (12, 24), "dpi": 80}

    fig, axs = plt.subplots(**univariate_params)

    for i, col in enumerate(predictor):
        sns.histplot(
            data=df,
            x=col,
            kde=True,
            hue="diagnosis",
            ax=axs[i % univariate_params["nrows"]][i // univariate_params["nrows"]],
            legend=False,
            palette={1: colors[0], 0: colors[3]},
        )
    plt.subplots_adjust(hspace=2)
    fig.suptitle("Breast Cancer Predictors Univariate Distribution", y=1.01, fontsize="x-large")
    fig.legend(df["diagnosis"].unique())
    fig.tight_layout()
    plt.show()


def plot_univariate_boxplot(df: pd.DataFrame, predictor: str) -> None:
    """Take in continuous predictors and plot univariate boxplot distribution. Note in this setting, we have kde=True.

    Args:
        df (pd.DataFrame): [description]
        predictor (str): [description]
    """

    univariate_params = {"nrows": 10, "ncols": 3, "figsize": (12, 24), "dpi": 80}

    fig, axs = plt.subplots(**univariate_params)

    for i, col in enumerate(predictor):
        sns.boxplot(
            data=df,
            x=col,
            hue="diagnosis",
            ax=axs[i % univariate_params["nrows"]][i // univariate_params["nrows"]],
        )
    plt.subplots_adjust(hspace=2)
    fig.suptitle("Breast Cancer Predictors Boxplot Distribution", y=1.01, fontsize="x-large")
    fig.legend(df["diagnosis"].unique())
    fig.tight_layout()
    plt.show()


def plot_heatmap(df: pd.DataFrame, predictors: List[str], cmap: str) -> pd.DataFrame:
    """This function takes in a dataframe and a list of predictors, and output the correlation matrix, as well as a plot of heatmap.

    1. Note that annot_kws attempts to make the size of the font visible and contained in the heatmap.
    2. Note that the CMAP is reversed and darker color indicates higher correlation as I find this more intuitive.

    Args:
        df (pd.DataFrame): [description]
        predictors (List[str]): [description]

    Returns:
        pd.DataFrame: [description]
    """

    corr = df[predictors].corr()
    annot_kws = {"size": 35 / np.sqrt(len(corr))}

    fig, _ = plt.subplots(figsize=(16, 12))
    sns.heatmap(corr, annot=True, cmap=cmap, annot_kws=annot_kws)
    return corr


def corrfunc(x: np.ndarray, y: np.ndarray, ax=None, **kws) -> None:
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f"{r:.1f}", xy=(0.7, 0.15), xycoords=ax.transAxes)


def plot_dimensional_reduction(df: pd.DataFrame, predictor_cols: List[str], colors: List[str]):
    """Plots PCA and TSNE for visualization of higher dimension to lower dimension.

    Args:
        df (pd.DataFrame): [description]
        predictor_cols (List[str]): [description]
        colors (List[str]): [description]
    """
    X_standardized = preprocessing.StandardScaler().fit_transform(df[predictor_cols])

    # Binary classification: we can set n components to 2 to better visualize all features in 2 dimensions
    pca = decomposition.PCA(n_components=2)
    pca_2d = pca.fit_transform(X_standardized)

    tsne = manifold.TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1500)
    tsne_2d = tsne.fit_transform(X_standardized)

    # Plot the TSNE and PCA visuals side-by-side
    plt.figure(figsize=(16, 11))
    plt.subplot(121)
    plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=df["diagnosis"], edgecolor="None", alpha=0.35)
    plt.colorbar()
    plt.title("PCA Scatter Plot")
    plt.subplot(122)
    plt.scatter(
        tsne_2d[:, 0],
        tsne_2d[:, 1],
        c=df["diagnosis"],
        edgecolor="None",
        alpha=0.35,
    )
    plt.colorbar()
    plt.title("TSNE Scatter Plot")
    plt.show()


def plot_precision_recall_vs_threshold(
    precisions: np.ndarray, recalls: np.ndarray, thresholds: np.ndarray
) -> None:
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.89
    and courtesy of https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc="best")


def plot_roc_curve(fpr, tpr, label=None):
    """
    The ROC curve, modified from
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91
    and courtesy of https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
    """
    plt.figure(figsize=(8, 8))
    plt.title("ROC Curve")
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0, 1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="best")
