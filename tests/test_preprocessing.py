import pandas as pd
from DS_package.data.preprocess import fill_missing, FillNa


def test_fill_missing_mean():
    df = pd.DataFrame({
        "Age": [10, 20, None]
    })

    result = fill_missing(df, "Age", FillNa.MEAN)

    assert result["Age"].isnull().sum() == 0
