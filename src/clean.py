import pandas as pd
from typing import List, Dict


def drop_columns(df: pd.DataFrame, columns: List) -> pd.DataFrame:
    """Drop unwanted columns

    Args:
        df (pd.DataFrame): [description]
        columns (List): [description]

    Returns:
        pd.DataFrame: [description]
    """

    df_copy = df.copy()
    df_copy = df_copy.drop(columns=columns, axis=1)
    return df_copy.reset_index(drop=True)


def class_mapping(df: pd.DataFrame, target: List[str], class_dict: Dict[str, int]) -> pd.DataFrame:
    """Takes in a dataframe and map the class from string to numbers.
    #TODO: Consider using Label Encoder.

    Args:
        df (pd.DataFrame): [description]
        target (List[str]): [description]
        class_dict (Dict[str, int]): [description]

    Returns:
        pd.DataFrame: [description]
    """

    df[target[0]] = df[target[0]].map(class_dict)
    return df


def report_missing(df: pd.DataFrame, columns: List) -> pd.DataFrame:
    """A function to check for missing data.

    Args:
        df (pd.DataFrame): [description]
        columns (List): [description]

    Returns:
        pd.DataFrame: [description]
    """
    missing_dict = {"missing num": [], "missing percentage": []}
    for col in columns:
        num_missing = df[col].isnull().sum()
        percentage_missing = num_missing / len(df)
        missing_dict["missing num"].append(num_missing)
        missing_dict["missing percentage"].append(percentage_missing)

    missing_data_df = pd.DataFrame(index=columns, data=missing_dict)

    return missing_data_df
