import pathlib
import pandas as pd
from src.database import write_to_sqlite3
from sklearn.preprocessing import LabelEncoder
from src.predict import (
    predict,
    cross_validate,
    PredictionParameters,
    SearchParameterSpace,
    search,
    postprocess_prediction,
    logistic_regression_predict,
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = pathlib.Path(__file__).parents[1] / "data"
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    return train_data, test_data


def split_train_data(train_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = train_data.drop(columns=["Survived"])
    y = train_data["Survived"]
    return x, y


def preprocess_data(
    train_x: pd.DataFrame, test_x: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Example preprocessing steps (replace with actual preprocessing logic)
    dropped_columns = ["PassengerId", "Name", "Ticket", "Cabin"]
    train_x = train_x.drop(columns=dropped_columns, errors="ignore")
    test_x = test_x.drop(columns=dropped_columns, errors="ignore")
    for c in ["Sex", "Embarked"]:
        le = LabelEncoder()
        train_x[c] = le.fit_transform(train_x[c].fillna("NA").astype(str))
        test_x[c] = le.transform(test_x[c].fillna("NA").astype(str))
    return train_x, test_x


def submission(prediction: pd.Series, test_raw: pd.DataFrame) -> None:
    submission_path = (
        pathlib.Path(__file__).parents[1] / "submission" / "submission.csv"
    )
    submission_df = pd.DataFrame(
        {"PassengerId": test_raw["PassengerId"], "Survived": prediction}
    )
    submission_df.to_csv(submission_path, index=False)


def main():
    train_data, test_x = load_data()
    test_raw = test_x.copy()
    train_x, train_y = split_train_data(train_data)
    write_to_sqlite3({"train": train_data, "test": test_x})
    train_x, test_x = preprocess_data(train_x, test_x)
    train_x_logistic = train_x.copy()
    train_y_logistic = train_y.copy()
    test_x_logistic = test_x.copy()
    params = PredictionParameters(random_state=71, n_estimators=20)
    param_space = SearchParameterSpace(
        max_depth=[3, 5, 6, 7], min_child_weight=[1.0, 2.0, 4.0]
    )
    params = search(train_x, train_y, params, param_space)
    print(f"Best parameters found: {params}")
    prediction = predict(train_x, train_y, test_x, params)
    cross_val_results = cross_validate(train_x, train_y, params)
    print(f"Cross-validation results: {cross_val_results}")
    print(train_x_logistic)
    print(train_y_logistic)
    print(test_x_logistic)
    logistic_prediction = logistic_regression_predict(
        train_x_logistic, train_y_logistic, test_x_logistic
    )
    print(f"Logistic regression predictions: {logistic_prediction}")
    total = 0.2 * logistic_prediction + 0.8 * prediction
    print(f"Ensemble predictions: {total}")
    postprocessed_prediction = postprocess_prediction(total)
    submission(postprocessed_prediction, test_raw)


if __name__ == "__main__":
    main()
