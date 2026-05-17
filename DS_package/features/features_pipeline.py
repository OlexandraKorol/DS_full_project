from DS_package.features import build_features , text_features
import pandas as pd
from DS_package.features import build_features, text_features
from sklearn.feature_extraction.text import TfidfVectorizer


def features_pipeline(df, columns_for_log_transform, text_column=None):
    """
    Full feature engineering pipeline that returns ONLY dataframe
    """

    df = build_features.create_family_size(df)
    df = build_features.create_is_alone(df)

    for col in columns_for_log_transform:
        df = build_features.log_transform(df, col)

    # -------------------------
    # 2. Text features (converted into df columns)
    # -------------------------
    if text_column and text_column in df.columns:
        text_series = df[text_column].fillna("")

        # TF-IDF
        vectorizer = TfidfVectorizer(max_features=50)  # зменшуємо розмір

        tfidf_matrix = vectorizer.fit_transform(text_series)

        # перетворюємо у df колонки
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
        )

        # додаємо до основного df
        df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

        # додаткові текстові фічі
        df = text_features.text_length(df, text_column)
        df = text_features.word_count(df, text_column)

    # -------------------------
    # return ONLY df
    # -------------------------
    return df
