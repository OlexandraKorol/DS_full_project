from DS_package.data import loader, preprocessing_pipeline, preprocess
from DS_package.features import features_pipeline
from models.models_pipeline import run_pipeline
import numpy as np


df = loader.load_data('DS_package/data/datasets/row/Titanic-Dataset.csv')

# delete Cabin - 77.104377 missing values

# [Age] - 19.865320
# [Embarked] - 0.224467

# можливо показати це візуально


#  preprocess

config = {
    "Age": preprocess.FillNa.MEAN,
    "Embarked": preprocess.FillNa.MODE,
}

clean_df = preprocessing_pipeline.preprocessing_pipeline(df, config)
loader.save_data(clean_df, "DS_package/data/datasets/processed/Titanic-Dataset-clean.csv")


# feature engeneering
print(clean_df.info())

log_columns = clean_df.select_dtypes(include=np.number).columns.tolist()
text_columns = ["Name", "Ticket", "Embarked"]
df = features_pipeline.features_pipeline(df, log_columns, text_columns)

print(df.info())


# model

result = run_pipeline(df, target="Survived")

print(result)