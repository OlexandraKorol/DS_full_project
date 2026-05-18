from DS_package.data import loader, preprocessing_pipeline, preprocess
from DS_package.features import features_pipeline
import numpy as np


df = loader.load_data('DS_package/data/datasets/raw/Titanic-Dataset.csv')

config = {
    "Age": preprocess.FillNa.MEAN,
    "Embarked": preprocess.FillNa.MODE,
}

categorical_cols = ["Sex", "Embarked"]

clean_df = preprocessing_pipeline.preprocessing_pipeline(df, config, categorical_cols)
loader.save_data(clean_df, "DS_package/data/datasets/processed/Titanic-Dataset-clean.csv")

log_columns = clean_df.select_dtypes(include=np.number).columns.tolist()
text_columns = ["Name", "Ticket", "Embarked"]

df = features_pipeline.features_pipeline(df, log_columns, text_columns)

# result = run_pipeline(df, target="Survived")
#
# print(result)
