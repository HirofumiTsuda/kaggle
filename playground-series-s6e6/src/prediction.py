import dataclasses

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import KFold
from xgboost import XGBClassifier


@dataclasses.dataclass(frozen=True)
class CrossValidationResults:
    accuracy: float
    logloss: float


@dataclasses.dataclass(frozen=True)
class PredictionParameters:
    random_state: int
    n_estimators: int
    output_column: str
    max_depth: int = 6
    min_child_weight: int = 1


def predict(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    params: PredictionParameters,
) -> tuple[np.ndarray, pd.Series]:
    model = XGBClassifier(n_estimators=params.n_estimators, random_state=params.random_state)
    model.fit(train_x, train_y)
    predictions = model.predict_proba(test_x)
    labels = pd.Series(np.argmax(predictions, axis=1), name=params.output_column)
    importance_df = pd.DataFrame(
        {
            "feature": train_x.columns,  # 学習に使ったDataFrameの列名
            "importance": model.feature_importances_,  # モデルが出した重要度
        }
    ).sort_values(by="importance", ascending=False)  # 重要度が高い順に並び替え
    print("feature importance:", importance_df)  # 上位10件を表示

    return predictions, labels


def cross_validate(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    params: PredictionParameters,
    param_span_splits: int = 4,
) -> CrossValidationResults:
    kfold = KFold(n_splits=param_span_splits, shuffle=True, random_state=params.random_state)
    accuracies = []
    loglosses = []

    for train_idx, val_idx in kfold.split(train_x):
        x_train, x_val = train_x.iloc[train_idx], train_x.iloc[val_idx]
        y_train, y_val = train_y.iloc[train_idx], train_y.iloc[val_idx]
        y_prod, y_label = predict(x_train, y_train, x_val, params)

        accuracies.append(accuracy_score(y_val, y_label))
        loglosses.append(log_loss(y_val, y_prod))

    return CrossValidationResults(accuracy=np.mean(accuracies), logloss=np.mean(loglosses))
