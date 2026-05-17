def general_data(df):
    return df.info()


def missing_report(df):
    return df.isnull().sum()



def missing_percent(df):
    return (df.isnull().mean() * 100).sort_values(ascending=False)



def basic_stats(df):
    return df.describe()


