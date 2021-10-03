from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn import (
    metrics,
    preprocessing,
    pipeline,
    tree,
    ensemble,
    linear_model,
    neighbors,
    dummy,
    svm,
)
from src import preprocess


# create a feature preparation pipeline for a model
def make_pipeline(model):
    """Make a Pipeline for Training.

    Args:
        model ([type]): [description]

    Returns:
        [type]: [description]
    """
    steps = list()
    # standardization
    steps.append(("standardize", preprocessing.StandardScaler()))
    # reduce VIF
    steps.append(("remove_multicollinearity", preprocess.ReduceVIF(thresh=10)))
    # the model
    steps.append(("model", model))
    # create pipeline
    _pipeline = pipeline.Pipeline(steps=steps)
    return _pipeline


def make_classifiers(seed: int) -> List[pipeline.Pipeline]:
    """Make Classifiers from make_pipeline function for rapid spot checking on Algorithms.

    Args:
        seed (int): [description]

    Returns:
        List[pipeline.Pipeline]: [description]
    """
    classifiers = [
        # baseline model
        dummy.DummyClassifier(random_state=seed, strategy="stratified"),
        # linear model
        linear_model.LogisticRegression(random_state=seed, solver="liblinear"),
        # nearest neighbours
        neighbors.KNeighborsClassifier(n_neighbors=8),
        # SVM
        svm.SVC(probability=True, random_state=seed),
        # tree
        tree.DecisionTreeClassifier(random_state=seed),
        # ensemble
        ensemble.RandomForestClassifier(random_state=seed),
    ]
    classifiers = [make_pipeline(model) for model in classifiers]
    return classifiers


def train_on_fold(
    df_folds: pd.DataFrame,
    models: List[Callable],
    num_folds: int,
    predictor_col: List,
    target_col: List,
) -> Dict[str, List]:
    """Take in a dataframe with fold number as column, and a models which holds a list of callable models, we will loop through and return a dictionary of cv results.

    Args:
        df_folds (pd.DataFrame): [description]
        model (Callable): [description]
        num_folds (int): [description]
        predictor_col (List): [description]
        target_col (List): [description]


    Returns:
        Dict[str, List]: [description]
    """

    y_true = df_folds[target_col].values.flatten()

    model_dict = {}

    for model in models:
        result_dict: Dict = {
            "identifier": [],
            "y_true": [],
            "y_pred": [],
            "y_prob": [],
            "brier_loss": [],
            "roc": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "confusion_matrix": [],
        }

        if isinstance(model, pipeline.Pipeline):
            model_name = model["model"].__class__.__name__
        else:
            model_name = model.__class__.__name__

        # out-of-fold validation predictions
        oof_pred_arr: np.ndarray = np.zeros(len(df_folds))

        for fold in range(1, num_folds + 1):

            train_df = df_folds[df_folds["fold"] != fold].reset_index(drop=True)
            val_df = df_folds[df_folds["fold"] == fold].reset_index(drop=True)
            val_idx = df_folds[df_folds["fold"] == fold].index.values
            X_train, y_train = train_df[predictor_col].values, train_df[target_col].values.ravel()
            X_val, y_val = val_df[predictor_col].values, val_df[target_col].values.ravel()

            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            y_val_prob = model.predict_proba(X_val)[:, 1]
            result_dict["y_true"].append(y_val)
            result_dict["y_pred"].append(y_val_pred)
            result_dict["y_prob"].append(y_val_prob)

            val_score = metrics.roc_auc_score(y_true=y_val, y_score=y_val_prob)
            precision, recall, fbeta_score, _ = metrics.precision_recall_fscore_support(
                y_true=y_val, y_pred=y_val_pred, labels=[0, 1], average=None
            )

            oof_pred_arr[val_idx] = y_val_pred.ravel()

            result_dict["identifier"].append(f"fold {fold}")
            result_dict["roc"].append(val_score)
            result_dict["precision"].append(precision)
            result_dict["recall"].append(recall)
            result_dict["f1"].append(fbeta_score)
            result_dict["brier_loss"].append(
                metrics.brier_score_loss(y_true=y_val, y_prob=y_val_prob)
            )
            result_dict["confusion_matrix"].append(metrics.confusion_matrix(y_val, y_val_pred))

        if model_name not in model_dict:
            model_dict[model_name] = result_dict

    return model_dict
