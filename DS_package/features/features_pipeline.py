from DS_package.features import build_features, text_features
import pandas as pd


def features_pipeline(df, log_columns, text_length_col=None, word_count_col=None, text_columns=None):
    """
       Full feature engineering pipeline for structured and text data.

       :param df: pandas.DataFrame
           Input dataset containing raw features.

       :param log_columns: list of str
           List of numerical columns to apply log transformation.

       :param  text_length_col: str, optional
           Name of the column used for creating text length feature.

       :param word_count_col: str, optional
           Name of the column used for creating word count feature.
           (Usually the same as text_length_col)

       :param text_columns: list of str, optional
           List of text columns to apply TF-IDF vectorization.

       :return: pandas.DataFrame enriched with:
           - dataset-specific features (FamilySize, IsAlone)
           - log-transformed numerical features
           - text statistics (length, word count)
           - TF-IDF features for selected text columns
       """

    df = build_features.create_family_size(df)
    df = build_features.create_is_alone(df)

    if text_length_col is not None:
        df = text_features.text_length(df, text_length_col)
    if text_length_col is not None:
        df = text_features.word_count(df, word_count_col)

    if log_columns:
        for col in log_columns:
            if col in df.columns:
                df = build_features.log_transform(df, col)

    for col in text_columns:
        if col in df.columns:

            tfidf_df = text_features.tfidf_features(df[col])

            df = pd.concat(
                [df.reset_index(drop=True),
                 tfidf_df.reset_index(drop=True)],
                axis=1
            )

    return df
