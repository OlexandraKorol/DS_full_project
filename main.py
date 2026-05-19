from DS_package.data import loader, preprocessing_pipeline, preprocess
from DS_package.features import features_pipeline
from DS_package.models import models_pipeline
import numpy as np


df = loader.load_data("DS_package/data/datasets/raw/Titanic-Dataset.csv")


y = df["Survived"].astype(int)
X = df.drop(columns=["Survived"])


config = {
    "Age": preprocess.FillNa.MEAN,
    "Embarked": preprocess.FillNa.MODE,
}

X = preprocessing_pipeline.preprocessing_pipeline(X, config, encode_objects=False)


log_columns = (X.select_dtypes(include=np.number).columns.tolist())

text_columns = ["Name", "Ticket"]

X = features_pipeline.features_pipeline(X, log_columns=log_columns, text_columns=text_columns)


loader.save_data(X, "DS_package/data/datasets/processed/Titanic-clean.csv")


result = models_pipeline.run_pipeline(X, y)

print(result)
