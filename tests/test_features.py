import pandas as pd

from DS_package.features.build_features import create_family_size
from DS_package.features.text_features import text_length, word_count


def test_create_family_size():

    df = pd.DataFrame({
        "SibSp": [1],
        "Parch": [2]
    })

    result = create_family_size(df)

    assert result["FamilySize"][0] == 4


def test_text_length():

    df = pd.DataFrame({
        "bio": ["hello world"]
    })

    result = text_length(df, "bio")

    assert result["bio_len"][0] == 11


def test_word_count():
    df = pd.DataFrame({
        "bio": ["hello beautiful world"]
    })

    result = word_count(df, "bio")

    assert result["bio_words"][0] == 3
