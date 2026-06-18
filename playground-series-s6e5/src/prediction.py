import dataclasses
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold

@dataclasses.dataclass(frozen=True)
class CrossValidationResults:
    accuracy: float
    logloss: float


@dataclasses.dataclass(frozen=True)
class PredictionParameters:
    random_state: int
    n_estimators: int
    output_column: str
    threshold: float = 0.5
    max_depth: int = 6
    min_child_weight: int = 1

def predict(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    params: PredictionParameters,
) -> tuple[pd.Series, pd.Series]:
    # mean for a driver
    group_by_key = ["Race"]
    print(train_x.columns)
    dropped_columns = [train_x.columns[i] for i in [0, 1, 4, 10, 12]]
    all_train = pd.concat([train_x, train_y], axis=1)
    means_df = (
        all_train.groupby(group_by_key)[params.output_column]
        .mean()
        .reset_index()
        .rename(columns={params.output_column: "RaceMean"})
        [group_by_key + ["RaceMean"]]
    )
    train_x = train_x.merge(means_df, on=group_by_key, how="left")
    train_x["RaceMean"] = train_x["RaceMean"].fillna(np.inf)
    test_x = test_x.merge(means_df, on=group_by_key, how="left")
    test_x["RaceMean"] = test_x["RaceMean"].fillna(np.inf)
    train_x = train_x.drop(columns=dropped_columns)
    test_x = test_x.drop(columns=dropped_columns)
    model = XGBClassifier(
        n_estimators=params.n_estimators, random_state=params.random_state
    )
    model.fit(train_x, train_y)
    result = model.predict_proba(test_x)
    predictions = pd.Series(result[:, 1], name=params.output_column)
    labels = predictions.apply(lambda x: 1 if x >= params.threshold else 0) 
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
    kfold = KFold(
        n_splits=param_span_splits, shuffle=True, random_state=params.random_state
    )
    accuracies = []
    loglosses = []

    for train_idx, val_idx in kfold.split(train_x):
        x_train, x_val = train_x.iloc[train_idx], train_x.iloc[val_idx]
        y_train, y_val = train_y.iloc[train_idx], train_y.iloc[val_idx]
        y_prod, y_label = predict(x_train, y_train, x_val, params)

        accuracies.append(accuracy_score(y_val, y_label))
        loglosses.append(log_loss(y_val, y_prod))

    return CrossValidationResults(
        accuracy=np.mean(accuracies), logloss=np.mean(loglosses)
    )
