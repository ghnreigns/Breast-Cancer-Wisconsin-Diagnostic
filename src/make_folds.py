from typing import List

import pandas as pd
from sklearn import model_selection


def make_folds(
    df: pd.DataFrame,
    num_folds: int,
    cv_schema: str,
    seed: int,
    predictor_col: List,
    target_col: List,
) -> pd.DataFrame:
    """Split the given dataframe into training folds.

    Args:
        df (pd.DataFrame): [description]
        num_folds (int): [description]
        cv_schema (str): [description]
        seed (int): [description]

    Returns:
        pd.DataFrame: [description]
    """

    if cv_schema == "KFold":
        df_folds = df.copy()
        kf = model_selection.KFold(n_splits=num_folds, shuffle=True, random_state=seed)

        for fold, (train_idx, val_idx) in enumerate(
            kf.split(X=df_folds[predictor_col], y=df_folds[target_col])
        ):
            df_folds.loc[val_idx, "fold"] = int(fold + 1)

        df_folds["fold"] = df_folds["fold"].astype(int)

    elif cv_schema == "StratifiedKFold":
        df_folds = df.copy()
        skf = model_selection.StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(X=df_folds[predictor_col], y=df_folds[target_col])
        ):
            df_folds.loc[val_idx, "fold"] = int(fold + 1)

        df_folds["fold"] = df_folds["fold"].astype(int)
        print(df_folds.groupby(["fold", "diagnosis"]).size())

    return df_folds
