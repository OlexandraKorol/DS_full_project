from DS_package.data import  preprocess
from DS_package.utils import utils

def preprocessing_pipeline(df, proceed_columns, missing_threshold=70, categorical_columns=None):

    """
    :param df: input dataframe
    :param proceed_columns: dict {column: FillNa strategy}
    :param missing_threshold: % threshold for dropping columns
    :return: processed dataframe
    """

    df = utils.drop_high_missing(df, threshold=missing_threshold)

    for column, strategy in proceed_columns.items():
        if column in df.columns:
            df = preprocess.fill_missing(df, column, strategy)


    df = preprocess.encode_categorical(df, categorical_columns)

    if "CreatedAt" in df.columns:
       df = preprocess.convert_dates(df, "CreatedAt")

    return df