import dataclasses

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
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


def xgboost_multi_balanced_accuracy(y_pred: np.ndarray, dtrain) -> tuple[str, float]:
    y_true = dtrain.get_label()

    y_pred_class = np.argmax(y_pred, axis=1)

    score = balanced_accuracy_score(y_true, y_pred_class)

    return "balanced_accuracy", score


def predict(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    params: PredictionParameters,
) -> tuple[np.ndarray, pd.Series]:
    model = XGBClassifier(
        n_estimators=params.n_estimators,
        random_state=params.random_state,
        objective="multi:softprob",
        eval_metric=xgboost_multi_balanced_accuracy,
    )
    train_sample_weights = compute_sample_weight(class_weight="balanced", y=train_y)
    model.fit(
        train_x,
        train_y,
        sample_weight=train_sample_weights,
    )
    predictions = model.predict_proba(test_x)
    labels = pd.Series(np.argmax(predictions, axis=1), name=params.output_column)
    importance_df = pd.DataFrame(
        {
            "feature": train_x.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)
    print("feature importance:", importance_df)

    return predictions, labels


def get_labels_from_thresholds(preds: np.ndarray, thresholds: list[float]) -> np.ndarray:
    n = len(preds[0, :])
    choices = [i + 1 for i in range(n)]
    result = np.dot(preds, choices)
    temp_thresholds = [-np.inf, *thresholds, np.inf]
    print("temp_thresholds:", temp_thresholds)  # Debugging line to check thresholds
    conditions = [
        (result >= temp_thresholds[i]) & (result < temp_thresholds[i + 1]) for i in range(n)
    ]
    final_labels = np.select(conditions, choices, default=-1)
    return final_labels


def objective_func(
    thresholds: np.ndarray,
    preds: np.ndarray,
    y_true: np.ndarray,
) -> float:

    labels = get_labels_from_thresholds(preds, thresholds)

    # scipyのminimizeは「最小化」を目指すため、
    # 最大化したい指標（Balanced Accuracy）にマイナスを付けて返す
    score = balanced_accuracy_score(y_true, labels)
    return -score


def post_predict(preds: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(preds[0, :])
    initial_weights = np.array([(i + 1) / n for i in range(n - 1)])

    # 最適化の実行
    result = minimize(
        fun=objective_func,
        x0=initial_weights,
        args=(preds, y_true),  # objective_funcに渡す追加の引数
        method="Nelder-Mead",  # 最適化アルゴリズムにNelder-Meadを指定
        options={"maxiter": 500},  # 最大イテレーション数（必要に応じて調整）
    )
    optimal_thresholds = result.x
    labels = get_labels_from_thresholds(preds, optimal_thresholds)

    return optimal_thresholds, labels


def cross_validate(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    params: PredictionParameters,
    param_span_splits: int = 4,
) -> CrossValidationResults:
    kfold = StratifiedKFold(
        n_splits=param_span_splits, shuffle=True, random_state=params.random_state
    )
    accuracies = []
    loglosses = []

    for train_idx, val_idx in kfold.split(train_x, train_y):
        x_train, x_val = train_x.iloc[train_idx], train_x.iloc[val_idx]
        y_train, y_val = train_y.iloc[train_idx], train_y.iloc[val_idx]
        y_prod, y_label = predict(x_train, y_train, x_val, params)

        accuracies.append(balanced_accuracy_score(y_val, y_label))
        loglosses.append(log_loss(y_val, y_prod))

    return CrossValidationResults(accuracy=np.mean(accuracies), logloss=np.mean(loglosses))
