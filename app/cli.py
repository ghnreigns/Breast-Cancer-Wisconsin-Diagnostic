import pandas as pd
import numpy as np
import csv
from sklearn import (
    base,
    ensemble,
    linear_model,
    metrics,
    model_selection,
    dummy,
    base,
    pipeline,
    preprocessing,
    svm,
    tree,
    neighbors,
    decomposition,
    feature_selection,
)
import pandas as pd
from typing import List, Dict, Union
import matplotlib.pyplot as plt
import random
from typing import Callable
import numpy as np
import pandas as pd
from scipy import stats
from functools import wraps
from time import time
import mlxtend
from mlxtend.evaluate import paired_ttest_5x2cv
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src import clean, train, make_folds, utils, eval
from config import global_params

CONFIG = global_params.global_config


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw data from a csv filepath.

    Args:
        filepath (str): [description]

    Returns:
        pd.DataFrame: [description]
    """
    try:
        df = pd.read_csv(filepath)
    except UnicodeDecodeError as e:
        print(e)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """[summary]

    Args:
        df (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """

    df_copy = df.copy()

    # Drop unwanted columns
    df_copy = clean.drop_columns(df, columns=CONFIG.unwanted_cols)

    # Map class
    df_copy = clean.class_mapping(df_copy, target=CONFIG.target, class_dict=CONFIG.class_dict)

    # Write to processed folder
    df_copy.to_csv(CONFIG.processed_final, index=False)

    return df_copy


def spot_checking(
    df: pd.DataFrame, classifiers: List[Callable]
) -> Union[pd.DataFrame, pd.DataFrame]:
    """This method is advocated by Jason Brownlee PhD and this serves as the first stage of my modelling process.
    We will rapidly test (spot check) different classifier algorithms.

    Args:
        df (pd.DataFrame): [description]
        classifiers (List[Callable]): [description]

    Returns:
        pd.DataFrame: [description]
    """

    X = df.copy()
    y = X.pop(CONFIG.target[0])
    predictor_cols = X.columns.to_list()
    # Split train - test
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X,
        y,
        train_size=CONFIG.split_size["train_size"],
        shuffle=True,
        stratify=y,
        random_state=CONFIG.seed,
    )
    X_y_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    df_folds = make_folds.make_folds(
        X_y_train,
        num_folds=CONFIG.num_folds,
        cv_schema=CONFIG.cv_schema,
        seed=CONFIG.seed,
        predictor_col=predictor_cols,
        target_col=CONFIG.target,
    )
    model_dict = train.train_on_fold(
        df_folds,
        models=classifiers,
        num_folds=5,
        predictor_col=predictor_cols,
        target_col=CONFIG.target,
    )

    for v in model_dict.values():
        utils.add_summary_col(v)

    results_df = pd.concat({k: pd.DataFrame(v).T for k, v in model_dict.items()}, axis=0)
    results_df.columns = ["fold 1", "fold 2", "fold 3", "fold 4", "fold 5", "mean_cv", "oof_cv"]
    results_df.to_csv(CONFIG.spot_checking, index=True)

    summary_df = eval.summarize_metrics(
        model_dict=model_dict,
        metric_name="roc",
        output_filepath=CONFIG.spot_checking_summary,
        output_imagepath=CONFIG.spot_checking_boxplot,
    )

    return results_df, summary_df


def grid_search(df: pd.DataFrame):
    X = df.copy()
    y = X.pop(CONFIG.target[0])
    predictor_cols = X.columns.to_list()
    # Split train - test
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X,
        y,
        train_size=CONFIG.split_size["train_size"],
        shuffle=True,
        stratify=y,
        random_state=CONFIG.seed,
    )

    pipeline_logistic = train.make_pipeline(
        linear_model.LogisticRegression(
            solver="saga", random_state=CONFIG.seed, max_iter=10000, n_jobs=None, fit_intercept=True
        )
    )
    param_grid = dict(
        # feature_selection__n_features_to_select=np.arange(5, 20, 2),
        model__penalty=["l1", "l2"],
        model__C=np.logspace(-4, 4, 10),
    )
    grid = model_selection.GridSearchCV(
        pipeline_logistic, param_grid=param_grid, cv=5, refit=True, verbose=3, scoring="roc_auc"
    )
    _ = grid.fit(X_train, y_train)
    grid_cv_df = pd.DataFrame(grid.cv_results_)
    return grid_cv_df.loc[grid_cv_df["rank_test_score"] == 1]


if __name__ == "__main__":
    df = load_data(filepath=CONFIG.raw_data)
    df = prepare_data(df)
    spot_check_df = spot_checking(df=df, classifiers=train.make_classifiers(CONFIG.seed))

    # print(spot_check_df)
    # grid_df = grid_search(df)
    # print(grid_df)
