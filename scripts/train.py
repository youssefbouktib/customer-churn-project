"""Train, evaluate, and save the reproducible churn model."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from churn_project.modeling import (  # noqa: E402
    FEATURES,
    RANDOM_STATE,
    TARGET,
    evaluate_predictions,
    load_and_clean_data,
    select_model_with_oof,
)

DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
ARTIFACTS = ROOT / "artifacts"


def churn_profile(frame: pd.DataFrame, column: str) -> list[dict]:
    grouped = frame.groupby(column, dropna=False)[TARGET].agg(rate="mean", customers="size")
    return [
        {"category": str(index), "churn_rate": float(row.rate), "customers": int(row.customers)}
        for index, row in grouped.iterrows()
    ]


def top_model_factors(pipeline) -> list[dict]:
    preprocessor = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    if hasattr(estimator, "coef_"):
        values = estimator.coef_[0]
        order = np.argsort(np.abs(values))[::-1][:12]
        return [
            {"feature": str(names[index]), "effect": "higher churn score" if values[index] > 0 else "lower churn score",
             "magnitude": float(abs(values[index]))}
            for index in order
        ]
    if hasattr(estimator, "estimators_"):
        importances = [
            fitted.feature_importances_ / max(fitted.feature_importances_.sum(), np.finfo(float).eps)
            for fitted in estimator.estimators_
            if hasattr(fitted, "feature_importances_")
        ]
        if importances:
            values = np.mean(importances, axis=0)
            order = np.argsort(values)[::-1][:12]
            return [
                {"feature": str(names[index]), "effect": "mean tree importance (direction not shown)",
                 "magnitude": float(values[index])}
                for index in order
            ]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        order = np.argsort(values)[::-1][:12]
        return [
            {"feature": str(names[index]), "effect": "importance (direction not shown)",
             "magnitude": float(values[index])}
            for index in order
        ]
    return []


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(
            f"Dataset not found at {DATA_PATH}. Add the Telco churn CSV there; "
            "see README.md for setup instructions."
        )

    data = load_and_clean_data(DATA_PATH)
    x = data[FEATURES]
    y = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    model_name, model, threshold, cv_results = select_model_with_oof(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, probabilities, threshold)

    # The published profile charts are descriptive summaries of the supplied dataset.
    profile_data = data.copy()
    tenure_labels = ["0–12", "13–24", "25–48", "49–72"]
    profile_data["TenureGroup"] = pd.cut(
        profile_data["tenure"], bins=[-1, 12, 24, 48, 72], labels=tenure_labels
    ).astype(str)

    metrics = {
        "project_title": "Customer Churn Risk Explorer",
        "model_name": model_name,
        "decision_threshold": threshold,
        "threshold_selection": "F1-maximizing threshold selected from 3-fold out-of-fold predictions on the training split.",
        "test_metrics": test_metrics,
        "cross_validation_comparison": cv_results,
        "dataset": {
            "rows": int(len(data)),
            "churn_rate": float(y.mean()),
            "positive_label": "Churn = Yes",
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "split": "80/20 stratified train/test split; random_state=42",
        },
        "profiles": {
            "Contract": churn_profile(profile_data, "Contract"),
            "InternetService": churn_profile(profile_data, "InternetService"),
            "PaymentMethod": churn_profile(profile_data, "PaymentMethod"),
            "TenureGroup": churn_profile(profile_data, "TenureGroup"),
        },
        "top_model_factors": top_model_factors(model),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, ARTIFACTS / "churn_pipeline.joblib")
    (ARTIFACTS / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"Selected model: {model_name}")
    print(f"Decision threshold (selected on training CV): {threshold:.3f}")
    print("Held-out test results:")
    for key in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        print(f"  {key}: {test_metrics[key]:.3f}")
    print(f"Artifacts saved to {ARTIFACTS}")


if __name__ == "__main__":
    main()

\n