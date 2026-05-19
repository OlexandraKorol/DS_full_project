from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def tfidf_features(text_series, max_features=100):

    text_series = text_series.fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        max_features=max_features
    )

    tfidf_matrix = vectorizer.fit_transform(text_series)

    return pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[
            f"tfidf_{i}"
            for i in range(tfidf_matrix.shape[1])
        ]
    )


def text_length(df, column):
    df = df.copy()

    df[f"{column}_len"] = (
        df[column]
        .fillna("")
        .astype(str)
        .apply(len)
    )

    return df


def word_count(df, column):
    df = df.copy()

    df[f"{column}_words"] = (
        df[column]
        .fillna("")
        .astype(str)
        .apply(lambda x: len(x.split()))
    )

    return df
