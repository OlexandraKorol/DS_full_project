# DS Package

Підсумкова практична робота з курсу Data Science in Python.

Проєкт реалізує повний цикл Data Science workflow: завантаження даних, preprocessing, feature engineering, feature selection, навчання моделей та оцінювання результатів.

Використано самостійно обраний датасет [wine-dataset](https://www.kaggle.com/datasets/elvinrustam/wine-dataset)

## Структура проєкту

```text
DS_package/
│
├── data/
│   ├── datasets/
│   │   ├── raw/
│   │   └── processed/
│   │
│   ├── loader.py
│   ├── preprocess.py
│   └── preprocessing_pipeline.py
│
├── features/
│   ├── build_features.py
│   ├── text_features.py
│   └── features_pipeline.py
│
├── models/
│   ├── train.py
│   ├── split_data.py
│   ├── selector.py
│   ├── evaluate.py
│   └── models_pipeline.py
│
├── utils/
│   ├── utils.py
│   └── validation.py
│
├── main.py
└── README.md
```

Структура побудована у вигляді Python-пакета із підпакетами для роботи з даними, ознаками та моделями. Для всіх папок додані файли `__init__.py`, що забезпечує коректну роботу імпортів.


## Робота з даними

Було створено підпакет `data`, який відповідає за завантаження, очищення та попередню підготовку даних.

Структура:

```text
data/
│
├── loader.py
├── preprocess.py
└── preprocessing_pipeline.py
```

### loader.py

Модуль використовується для:

- завантаження датасету у DataFrame;
- збереження оброблених даних.

Приклад:

```python
df = loader.load_data(path)
loader.save_data(df, save_path)
```

### preprocess.py

Містить функції для базової обробки:

- заповнення пропущених значень;
- one-hot кодування;
- обробки дат.

Реалізовано Enum `FillNa`:

```python
class FillNa(Enum):
    MEDIAN
    MODE
    MEAN
```

що дозволяє задавати стратегію очищення для кожної колонки.

Наприклад:

```python
config = {
    "Age": FillNa.MEAN,
    "Embarked": FillNa.MODE
}
```

### preprocessing_pipeline.py

Створено окремий pipeline, який автоматизує preprocessing:

1. видаляє колонки з великою кількістю пропусків;
2. застосовує задані стратегії заповнення;
3. виконує one-hot encoding;
4. перетворює дати. (Конвертація дат застосовується лише за наявності datetime-колонок)

Це дозволяє централізовано керувати preprocessing-етапом.


