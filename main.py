from DS_package.data import loader, pipeline, preprocessing_pipeline
from DS_package.features import features_pipeline, text_features

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

clean_df = pipeline.preprocessing_pipeline(df, config)
loader.save_data(clean_df, "DS_package/data/datasets/processed/Titanic-Dataset-clean.csv")


# feature engeneering
print(clean_df.info())

# vector = text_features.tfidf_features(['Ticket'])
# print(vector.X)
