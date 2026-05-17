from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def tfidf_features(text_series, max_features=500):
    vectorizer = TfidfVectorizer(max_features=max_features)

    tfidf_matrix = vectorizer.fit_transform(text_series)

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    )
    return tfidf_df

def text_length(df, column):
    df[column + "_len"] = df[column].apply(len)
    return df

def word_count(df, column):
    df[column + "_words"] = df[column].apply(lambda x: len(x.split()))
    return df


