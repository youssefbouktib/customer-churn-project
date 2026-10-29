"""Data preparation, model selection, and evaluation for churn prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

TARGET = "Churn"
ID_COLUMN = "customerID"
RAW_NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
ENGINEERED_NUMERIC_FEATURES = [
    "AvgMonthlySpend", "TenureYears", "TotalServices",
    "Risk_MonthToMonth_ECheck", "Risk_SuperCombo", "Risk_HighInitialCost",
]
NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = RAW_NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create the behavioral features using training-fitted statistics only."""

    def fit(self, x: pd.DataFrame, y=None):
        self.monthly_charge_median_ = float(
            pd.to_numeric(x["MonthlyCharges"], errors="coerce").median()
        )
        self.feature_names_in_ = np.asarray(x.columns, dtype=object)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        frame = x.copy()
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
        frame["AvgMonthlySpend"] = frame["TotalCharges"] / (frame["tenure"] + 1)
        frame["TenureYears"] = frame["tenure"] / 12.0

        service_columns = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies",
        ]
        frame["TotalServices"] = frame[service_columns].eq("Yes").sum(axis=1)
        month_to_month = frame["Contract"].eq("Month-to-month").astype(int)
        electronic_check = frame["PaymentMethod"].eq("Electronic check").astype(int)
        fiber_optic = frame["InternetService"].eq("Fiber optic").astype(int)
        frame["Risk_MonthToMonth_ECheck"] = month_to_month * electronic_check
        frame["Risk_SuperCombo"] = month_to_month * electronic_check * fiber_optic
        frame["Risk_HighInitialCost"] = (
            frame["tenure"].le(6)
            & frame["MonthlyCharges"].gt(self.monthly_charge_median_)
        ).astype(int)
        return frame

    def get_feature_names_out(self, input_features=None):
        names = list(input_features if input_features is not None else self.feature_names_in_)
        return np.asarray(names + ENGINEERED_NUMERIC_FEATURES, dtype=object)


def load_and_clean_data(path: str | Path) -> pd.DataFrame:
    """Load the Telco CSV and normalize the target and charge columns."""
    frame = pd.read_csv(path)
    required = set(FEATURES + [TARGET])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    frame = frame.copy()
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame[TARGET] = frame[TARGET].map({"No": 0, "Yes": 1})
    frame = frame.dropna(subset=[TARGET]).reset_index(drop=True)
    frame[TARGET] = frame[TARGET].astype("int8")
    return frame


def make_pipeline(model: Any) -> Pipeline:
    """Build preprocessing and estimator as one deployable object."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("feature_engineering", ChurnFeatureEngineer()),
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


def candidate_models(y_train: pd.Series | None = None) -> dict[str, Any]:
    """Return transparent baselines and the original boosting stack."""
    positive_weight = 2.8
    if y_train is not None:
        positives = max(int((y_train == 1).sum()), 1)
        positive_weight = float((y_train == 0).sum() / positives)

    stack = StackingClassifier(
        estimators=[
            (
                "xgb",
                XGBClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.01,
                    subsample=0.6, colsample_bytree=0.6,
                    scale_pos_weight=positive_weight, random_state=RANDOM_STATE,
                    eval_metric="logloss", n_jobs=1,
                ),
            ),
            (
                "cat",
                CatBoostClassifier(
                    iterations=300, depth=6, learning_rate=0.03,
                    auto_class_weights="Balanced", random_seed=RANDOM_STATE,
                    verbose=0, thread_count=1,
                ),
            ),
            (
                "lgb",
                LGBMClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.02,
                    class_weight="balanced", random_state=RANDOM_STATE,
                    verbose=-1, n_jobs=1,
                ),
            ),
        ],
        final_estimator=LogisticRegression(C=0.1, random_state=RANDOM_STATE, max_iter=2000),
        cv=3,
        n_jobs=1,
    )
    return {
        "Logistic regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost + CatBoost + LightGBM stack": stack,
    }


def best_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    """Choose the F1-maximizing threshold from validation predictions."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5, float(f1_score(y_true, probabilities >= 0.5, zero_division=0))
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], np.finfo(float).eps)
    )
    index = int(np.argmax(f1_values))
    return float(thresholds[index]), float(f1_values[index])


def evaluate_predictions(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "support": int(len(y_true)),
    }


def select_model_with_oof(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[str, Pipeline, float, list[dict]]:
    """Select an estimator using training-only out-of-fold probabilities."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    results = []
    models = candidate_models(y_train)
    for name, estimator in models.items():
        print(f"Evaluating {name} with 3-fold training-only cross-validation...", flush=True)
        pipeline = make_pipeline(estimator)
        oof_probabilities = cross_val_predict(
            pipeline, x_train, y_train, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        threshold, tuned_f1 = best_f1_threshold(y_train, oof_probabilities)
        results.append(
            {
                "name": name,
                "oof_f1_at_0_5": float(f1_score(y_train, oof_probabilities >= 0.5, zero_division=0)),
                "oof_f1_at_selected_threshold": tuned_f1,
                "selected_threshold": threshold,
            }
        )

    winner = max(results, key=lambda row: row["oof_f1_at_selected_threshold"])
    final_pipeline = make_pipeline(models[winner["name"]])
    final_pipeline.fit(x_train, y_train)
    return winner["name"], final_pipeline, winner["selected_threshold"], results

\n