import numpy as np

def create_family_size(df):
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df


def create_is_alone(df):
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def log_transform(df, column):
    df[column] = np.log1p(df[column])
    return df


