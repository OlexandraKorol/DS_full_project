from DS_package.models.split_data import split_data
from DS_package.models.selector import select_best_model


def run_pipeline(X, y):

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(X, y)

    name, model, score = select_best_model(
        X_train,
        X_test,
        y_train,
        y_test
    )

    return {
        "best_model_name": name,
        "best_model": model,
        "f1_score": score
    }
