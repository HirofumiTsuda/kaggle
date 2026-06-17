import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import dataclasses
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold
from typing import TypedDict
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer


@dataclasses.dataclass(frozen=True)
class CrossValidationResults:
    accuracy: float
    logloss: float


@dataclasses.dataclass(frozen=True)
class PredictionParameters:
    random_state: int
    n_estimators: int
    max_depth: int = 6
    min_child_weight: int = 1


class SearchParameterSpace(TypedDict):
    max_depth: list[int]
    min_child_weight: list[int]


def cross_validate(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    params: PredictionParameters,
    param_span_splits: int = 4,
) -> CrossValidationResults:
    kfold = KFold(
        n_splits=param_span_splits, shuffle=True, random_state=params.random_state
    )
    accuracies = []
    loglosses = []

    for train_idx, val_idx in kfold.split(train_x):
        x_train, x_val = train_x.iloc[train_idx], train_x.iloc[val_idx]
        y_train, y_val = train_y.iloc[train_idx], train_y.iloc[val_idx]

        model = XGBClassifier(
            n_estimators=params.n_estimators,
            random_state=params.random_state,
            max_depth=params.max_depth,
            min_child_weight=params.min_child_weight,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        y_prob = model.predict_proba(x_val)[:, 1]

        accuracies.append(accuracy_score(y_val, y_pred))
        loglosses.append(log_loss(y_val, y_prob))

    return CrossValidationResults(
        accuracy=np.mean(accuracies), logloss=np.mean(loglosses)
    )


def predict(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    params: PredictionParameters,
) -> pd.Series:

    model = XGBClassifier(
        n_estimators=params.n_estimators, random_state=params.random_state
    )
    model.fit(train_x, train_y)
    result = model.predict_proba(test_x)
    predictions = result[:, 1]

    return pd.Series(predictions, name="Survived")


def search(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    params: PredictionParameters,
    override_params: SearchParameterSpace,
    param_span_splits: int = 4,
) -> PredictionParameters:
    param_combinations = itertools.product(
        override_params["max_depth"], override_params["min_child_weight"]
    )
    candidate_params = []
    candidate_scores = []
    for max_depth, min_child_weight in param_combinations:
        current_params = PredictionParameters(
            random_state=params.random_state,
            n_estimators=params.n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
        )
        cv_results = cross_validate(train_x, train_y, current_params, param_span_splits)
        candidate_params.append(current_params)
        candidate_scores.append(cv_results)

    best_index = np.argmin([score.logloss for score in candidate_scores])
    return candidate_params[best_index]


def postprocess_prediction(prediction: pd.Series) -> pd.Series:
    return prediction.apply(lambda x: 1 if x > 0.5 else 0)


def logistic_regression_predict(
    train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame
) -> pd.Series:
    pipeline = Pipeline(
        [
            ("imputer", KNNImputer(n_neighbors=5)),
            ("classifier", LogisticRegression(solver="lbfgs", max_iter=300)),
        ]
    )
    pipeline.fit(train_x, train_y)
    predictions = pipeline.predict_proba(test_x)[:, 1]
    return pd.Series(predictions, name="Survived")
