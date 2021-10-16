from typing import Dict, Union, Callable
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def summarize_metrics(
    model_dict: Dict, metric_name: str, output_filepath: str, output_imagepath: str
) -> pd.DataFrame:
    """Take in the model dict output from spot checking and define a metric name and save all folds results for the metric and it's standard error.
    We will save the dataframe and also plot a boxplot for the performance across various models.

    Args:
        metric_name (str): [description]
        model_dict (Dict): [description]

    Returns:
        [type]: [description]
    """
    ls = []
    for model_name, inner_dict in model_dict.items():
        folds = inner_dict["identifier"][:-2]
        all_obs = []
        for idx, obs in enumerate(inner_dict[metric_name][:-2]):
            ls.append((model_name, folds[idx], obs))
            all_obs.append(obs)
        ls.append((model_name, "SE", np.std(all_obs, ddof=1) / len(all_obs) ** 0.5))

    fig, ax = plt.subplots(figsize=(15, 8))

    summary_df = pd.DataFrame(ls, columns=["model", "fold", metric_name])
    summary_df.to_csv(output_filepath, index=False)

    _ = sns.boxplot(
        x="model",
        y=metric_name,
        data=summary_df[(summary_df["model"] != "DummyClassifier") & (summary_df["fold"] != "SE")],
        ax=ax,
    )

    fig.savefig(output_imagepath, format="png", dpi=300)

    return summary_df


def evaluate_train_test_set(
    estimator: Callable, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
) -> Dict[str, Union[float, np.ndarray]]:
    """This function takes in X and y and returns a dictionary of scores.

    Args:
        estimator (Callable): [description]
        X (Union[pd.DataFrame, np.ndarray]): [description]
        y (Union[pd.DataFrame, np.ndarray]): [description]

    Returns:
        Dict[str, Union[float, np.ndarray]]: [description]
    """

    test_results = {}

    y_pred = estimator.predict(X)
    # This is the probability array of class 1 (malignant)
    y_prob = estimator.predict_proba(X)[:, 1]

    test_brier = metrics.brier_score_loss(y, y_prob)
    test_roc = metrics.roc_auc_score(y, y_prob)

    test_results["brier"] = test_brier
    test_results["roc"] = test_roc
    test_results["y"] = np.asarray(y).flatten()
    test_results["y_pred"] = y_pred.flatten()
    test_results["y_prob"] = y_prob.flatten()

    return test_results
