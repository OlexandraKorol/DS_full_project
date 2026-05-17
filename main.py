from DS_package.data import loader
from DS_package.utils.utils import general_data, missing_report, missing_percent, basic_stats

df = loader.load_data('DS_package/data/datasets/row/Titanic-Dataset.csv')

# delete Cabin - 77.104377 missing values

# [Age] - 19.865320
# [Embarked] - 0.224467

print(missing_percent(df))

# можливо показати це візуально





