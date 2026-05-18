from models.train import models
from models.evaluate import evaluate

def select_best_model(
        X_train,
        X_test,
        y_train,
        y_test
):

    best_model = None
    best_name = ""
    best_score = 0

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        score = evaluate(
            model,
            X_test,
            y_test
        )

        print(
            f"{name}: {score:.3f}"
        )

        if score > best_score:

            best_score = score
            best_model = model
            best_name = name

    return (
        best_name,
        best_model,
        best_score
    )