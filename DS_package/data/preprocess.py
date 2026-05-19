from enum import Enum
import pandas as pd


class FillNa(Enum):
    MEDIAN = "median"
    MODE = "mode"
    MEAN = "mean"


def fill_missing(df, column, strategy: FillNa):
    """
    Fills missing values in a dataframe column
    using the selected strategy.

    :param df: Input dataframe.
    :param column: Name of the column to process.
    :param strategy: FillNa
        Missing value filling strategy.
        Available options:
        - FillNa.MEAN
        - FillNa.MEDIAN
        - FillNa.MODE

    :return: Dataframe with filled missing values.
    """
    df = df.copy()

    if strategy == FillNa.MEAN:
        value = df[column].mean()

    elif strategy == FillNa.MEDIAN:
        value = df[column].median()

    elif strategy == FillNa.MODE:
        value = df[column].mode()[0]

    else:
        raise ValueError("Unknown fill strategy")

    df[column] = df[column].fillna(value)

    return df


def encode_all_objects(df):
    df = df.copy()

    obj_cols = df.select_dtypes(include=["object"]).columns

    return pd.get_dummies(
        df,
        columns=obj_cols,
        drop_first=True
    )


def convert_dates(df, column):
    df = df.copy()

    df[column] = pd.to_datetime(df[column])

    df[f"{column}_year"] = df[column].dt.year
    df[f"{column}_month"] = df[column].dt.month
    df[f"{column}_day"] = df[column].dt.day

    df = df.drop(columns=[column])

    return df
