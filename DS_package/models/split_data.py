from sklearn.model_selection import train_test_split


def split_data(df, target):

    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
