# DS Package

Підсумкова практична робота з курсу Data Science in Python.
Проєкт реалізує повний цикл Data Science workflow: завантаження даних, preprocessing, feature engineering, feature selection, навчання моделей та оцінювання результатів.

Використано самостійно обраний датасет [Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

## Запуск проєкту

### Встановлення залежностей

```bash
pip install -r requirements.txt
```

### Запуск main.py

```bash
python main.py
```


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
├── tests/
│   ├── test_preprocessing.py
│   └── test_features.py
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


## Feature Engineering

Для створення нових ознак було реалізовано підпакет `features`.

Структура:

```text
features/
│
├── build_features.py
├── text_features.py
└── features_pipeline.py
```

### build_features.py

Було створено нові ознаки:

- `FamilySize` — кількість членів сім'ї пасажира:

```python
FamilySize = SibSp + Parch + 1
```

- `IsAlone` — бінарна ознака, яка показує, чи подорожував пасажир самостійно:

```python
IsAlone = 1 якщо FamilySize == 1
```

Такі ознаки можуть бути корисними, оскільки соціальний фактор може впливати на виживання.

Також реалізовано логарифмічну трансформацію:

```python
np.log1p()
```

що використовується для зменшення асиметрії числових ознак.

### text_features.py

Для текстових ознак створено:

- довжину тексту;
- кількість слів;
- TF-IDF представлення.

TF-IDF дозволяє перетворити текст у числовий формат для подальшого використання моделями машинного навчання.

### features_pipeline.py

Усі кроки feature engineering були об'єднані у єдиний pipeline:

1. створення нових ознак;
2. створення текстових характеристик;
3. логарифмічне перетворення;
4. TF-IDF векторизація.


## Побудова моделей

Було створено підпакет `models`, який реалізує повний цикл навчання та оцінювання моделей.

Структура:

```text
models/
│
├── train.py
├── split_data.py
├── selector.py
├── evaluate.py
└── models_pipeline.py
```

### train.py

Було реалізовано набір моделей:

- kNN
- Naive Bayes
- Logistic Regression

Для моделей, чутливих до масштабу ознак, використано:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", ...)
])
```

Масштабування є важливим для моделей, що працюють із відстанями або у лінійному просторі.

### split_data.py

Для розділення даних використано:

```python
train_test_split()
```

Дані були розділені у співвідношенні:

- train: 80%
- test: 20%

Для уникнення дисбалансу класів використано:

```python
stratify=y
```

### evaluate.py

Для оцінювання використано метрику:

- F1-score

### selector.py

Було реалізовано автоматичний перебір моделей.

Алгоритм:

1. навчити модель;
2. отримати прогноз;
3. обчислити F1;
4. порівняти результат;
5. зберегти найкращу модель.

### models_pipeline.py

Pipeline автоматизує повний цикл:

```text
DataFrame
 ↓
split
 ↓
навчання
 ↓
оцінювання
 ↓
вибір кращої моделі
```




tests  python -m pytest
flake8
EDA
feature selection

# Тестування

Для тестування проєкту використано `pytest`.

Було написано unit tests для:

* preprocessing функцій;
* feature engineering функцій.

## Запуск тестів

```bash
python -m pytest
```

Результат:

```text
===================== test session starts =====================
platform darwin -- Python 3.14.0rc2, pytest-9.0.3, pluggy-1.6.0
rootdir: /Users/Sandra/PycharmProjects/DS_Project
collected 4 items                                                                                                                                                          

tests/test_features.py ...                                                                                                                                           [ 75%]
tests/test_preprocessing.py .  

===================== 4 passed =================================
```

---


