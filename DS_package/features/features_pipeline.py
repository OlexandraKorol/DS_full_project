from DS_package.features import build_features, text_features
from DS_package.data import preprocess
import pandas as pd


def features_pipeline(df, log_columns=None, text_columns=None):
    df = df.copy()

    if log_columns is None:
        log_columns = []

    if text_columns is None:
        text_columns = []


    if {"SibSp", "Parch"}.issubset(df.columns):

        df = build_features.create_family_size(df)

        df = build_features.create_is_alone(df)


    for col in text_columns:

        if col not in df.columns:
            continue

        df = text_features.text_length(df, col)

        df = text_features.word_count(df, col)

        tfidf_df = text_features.tfidf_features(df[col])

        df = pd.concat(
            [
                df.reset_index(drop=True),
                tfidf_df.reset_index(drop=True)
            ],
            axis=1
        )

        df = df.drop(columns=[col])


    for col in log_columns:

        if col in df.columns:

            if (df[col] >= 0).all():

                df = build_features.log_transform(
                    df,
                    col
                )


    df = preprocess.encode_all_objects(df)

    return df
