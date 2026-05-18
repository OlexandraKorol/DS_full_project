from DS_package.data import preprocess
from DS_package.utils import utils


def preprocessing_pipeline(df, proceed_columns, categorical_columns=None):

    """
    :param categorical_columns:
    :param df: input dataframe
    :param proceed_columns: dict {column: FillNa strategy}
    :return: processed dataframe
    """

    df = utils.drop_high_missing(df, threshold=70)

    for column, strategy in proceed_columns.items():
        if column in df.columns:
            df = preprocess.fill_missing(df, column, strategy)

    df = preprocess.encode_categorical(df, categorical_columns)

    if "CreatedAt" in df.columns:
        df = preprocess.convert_dates(df, "CreatedAt")

    return df
