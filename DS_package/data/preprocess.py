from enum import Enum
import pandas as pd


class FillNa(Enum):
    MEDIAN = "median"
    MODE = "mode"
    MEAN = "mean"

def fill_missing(df, column, strategy: FillNa):

    """
    :param df:  df
    :param column: column to destroy na
    :param strategy: how to destroy. Avaible options: FillNa.MEDIAN, FillNa.MODE, FillNa.MEAN
    :return: df
    """

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


def encode_categorical(df, columns):
    df = pd.get_dummies(df, columns=columns)
    return df


def convert_dates(df, column):
    df[column] = pd.to_datetime(df[column])
    df[f"{column}_year"] = df[column].dt.year
    return df
