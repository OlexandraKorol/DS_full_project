from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(text_series, max_features=500):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(text_series)
    return X, vectorizer

def text_length(df, column):
    df[column + "_len"] = df[column].apply(len)
    return df

def word_count(df, column):
    df[column + "_words"] = df[column].apply(lambda x: len(x.split()))
    return df


