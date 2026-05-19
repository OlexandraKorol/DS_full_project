from DS_package.data import preprocess
from DS_package.utils import utils


def preprocessing_pipeline(df, proceed_columns, encode_objects=False):
    """
    Full preprocessing pipeline for tabular data.

    :param df:  Input dataframe to preprocess.
    :param proceed_columns: dict
        Dictionary with fill strategies for columns.
        Format:
        {
            "column_name": FillNa.MEAN,
            "other_column": FillNa.MODE
        }
    :param encode_objects: bool, optional
        If True, applies one-hot encoding to all object columns.
        Default is False.

    :return:  Preprocessed dataframe.
    """
    df = df.copy()

    df = utils.drop_high_missing(df, threshold=70)

    for column, strategy in proceed_columns.items():
        if column in df.columns:
            df = preprocess.fill_missing(
                df,
                column,
                strategy
            )

    if "CreatedAt" in df.columns:
        df = preprocess.convert_dates(
            df,
            "CreatedAt"
        )

    if encode_objects:
        df = preprocess.encode_all_objects(df)

    return df
