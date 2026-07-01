import pathlib

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.prediction import (
    PredictionParameters,
    cross_validate,
    predict,
)

TARGET_COLUMN = "health_condition"
LABEL_COLUMNS = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = pathlib.Path(__file__).parents[1] / "data"
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    return train_data, test_data


def split_train_data(train_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = train_data.drop(columns=[TARGET_COLUMN])
    y = train_data[TARGET_COLUMN]
    return x, y


def preprocess_data(
    train_x: pd.DataFrame, test_x: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
    train_x.drop(columns=["id"], inplace=True)
    test_x.drop(columns=["id"], inplace=True)
    for c in LABEL_COLUMNS:
        le = LabelEncoder()
        train_x[c] = le.fit_transform(train_x[c].fillna("NA").astype(str))
        test_x[c] = le.transform(test_x[c].fillna("NA").astype(str))
    # For target column, we need to encode it as well
    le_target = LabelEncoder()
    train_x[TARGET_COLUMN] = le_target.fit_transform(
        train_x[TARGET_COLUMN].fillna("NA").astype(str)
    )
    return train_x, test_x, le_target


def postprocess_predictions(
    predictions: pd.Series, test_raw: pd.DataFrame, le_target: LabelEncoder
) -> pd.DataFrame:
    decoded_predictions = predictions.map(
        lambda x: le_target.inverse_transform([x])[0]
    )  # Decode labels back to original
    return pd.DataFrame({"id": test_raw["id"], TARGET_COLUMN: decoded_predictions})


def submission(result: pd.DataFrame) -> None:
    submission_path = pathlib.Path(__file__).parents[1] / "submissions" / "submission.csv"
    result.to_csv(submission_path, index=False)


def main():
    train_x, test_x = load_data()
    test_raw = test_x.copy()
    train_x, test_x, le_target = preprocess_data(train_x, test_x)
    train_x, train_y = split_train_data(train_x)
    params = PredictionParameters(
        random_state=42,
        n_estimators=100,
        output_column=TARGET_COLUMN,
    )
    predictions, labels = predict(train_x, train_y, test_x, params)
    validation_results = cross_validate(train_x, train_y, params)
    print("Predictions:", predictions)
    print("Cross-validation results:", validation_results)
    result = postprocess_predictions(labels, test_raw, le_target)
    submission(result)


if __name__ == "__main__":
    main()
