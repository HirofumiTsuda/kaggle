import pathlib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.prediction import (
    predict,
    cross_validate,
    PredictionParameters,
)

TARGET_COLUMN = "PitNextLap"

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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for c in ["Driver", "Compound", "Race"]:
        le = LabelEncoder()
        train_x[c] = le.fit_transform(train_x[c].fillna("NA").astype(str))
        test_x[c] = le.transform(test_x[c].fillna("NA").astype(str))
    return train_x, test_x


def submission(prediction: pd.Series, test_raw: pd.DataFrame) -> None:
    submission_path = (
        pathlib.Path(__file__).parents[1] / "submission" / "submission.csv"
    )
    submission_df = pd.DataFrame(
        {"id": test_raw["id"], TARGET_COLUMN: prediction}
    )
    submission_df.to_csv(submission_path, index=False)

def main():
    train_data, test_x = load_data()
    test_raw = test_x.copy()
    train_x, train_y = split_train_data(train_data)
    train_x, test_x = preprocess_data(train_x, test_x)
    params = PredictionParameters(
        random_state=42,
        n_estimators=100,
        output_column=TARGET_COLUMN,
    )
    predictions, _ = predict(train_x, train_y, test_x, params)
    validation_results = cross_validate(train_x, train_y, params)
    print("Predictions:", predictions.head())
    print("Cross-validation results:", validation_results)
    submission(predictions, test_raw)
    


if __name__ == "__main__":
    main()
