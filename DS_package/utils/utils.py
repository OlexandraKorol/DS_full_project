def missing_report(df):
    return df.isnull().sum()


def drop_high_missing(df, threshold=70):
    missing = df.isnull().mean() * 100
    cols_to_drop = missing[missing > threshold].index

    df = df.drop(columns=cols_to_drop)

    return df
