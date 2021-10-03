import matplotlib.pyplot as plt
import pandas as pd
import plotly.figure_factory as ff
import seaborn as sns
from typing import List
import numpy as np


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


def plot_distribution(df: pd.DataFrame, predictor: str, size_bin: float) -> None:
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        predictor (str): [description]
        size_bin (float): [description]
    """
    df_m = df.loc[df["diagnosis"] == 1]
    df_b = df.loc[df["diagnosis"] == 0]

    hist_data = [df_m[predictor], df_b[predictor]]

    group_labels = ["malignant", "benign"]
    colors = ["#FF0000", "#00FF00"]

    fig = ff.create_distplot(
        hist_data, group_labels, colors=colors, show_hist=True, bin_size=size_bin, curve_type="kde"
    )

    fig["layout"].update(title=predictor)

    py.iplot(fig, filename="Density plot")


def plot_univariate(df: pd.DataFrame, predictor: str) -> None:
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
            palette={1: "red", 0: "green"},
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
            kde=True,
            hue="diagnosis",
            ax=axs[i % univariate_params["nrows"]][i // univariate_params["nrows"]],
            legend=False,
            palette={1: "red", 0: "green"},
        )
    plt.subplots_adjust(hspace=2)
    fig.suptitle("Breast Cancer Predictors Univariate Distribution", y=1.01, fontsize="x-large")
    fig.legend(df["diagnosis"].unique())
    fig.tight_layout()
    plt.show()


def plot_heatmap(df: pd.DataFrame, predictors: List[str]) -> pd.DataFrame:
    """This function takes in a dataframe and a list of predictors, and output the correlation matrix, as well as a plot of heatmap.

    Note that annot_kws attempts to make the size of the font visible and contained in the heatmap.

    Args:
        df (pd.DataFrame): [description]
        predictors (List[str]): [description]

    Returns:
        pd.DataFrame: [description]
    """

    corr = df[predictors].corr()
    annot_kws = {"size": 35 / np.sqrt(len(corr))}

    fig, _ = plt.subplots(figsize=(16, 12))
    sns.heatmap(corr, annot=True, annot_kws=annot_kws)

    return corr


def corrfunc(x: np.ndarray, y: np.ndarray, ax=None, **kws) -> None:
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f"ρ = {r:.2f}", xy=(0.1, 0.95), xycoords=ax.transAxes)
