import numpy as np
import pandas as pd
import unittest
import tempfile
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from churn_project.modeling import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    RAW_NUMERIC_FEATURES,
    TARGET,
    ChurnFeatureEngineer,
    best_f1_threshold,
    load_and_clean_data,
    make_pipeline,
)


def sample_rows() -> pd.DataFrame:
    rows = []
    for index in range(24):
        churn = index % 2
        phone = "No" if index % 4 == 0 else "Yes"
        internet = "No" if index % 3 == 0 else ("DSL" if index % 2 == 0 else "Fiber optic")
        service_values = (
            {name: "No internet service" for name in (
                "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies",
            )}
            if internet == "No"
            else {
                name: ["No", "Yes"][index % 2]
                for name in (
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                    "StreamingTV", "StreamingMovies",
                )
            }
        )
        row = {
            "customerID": f"C{index:04d}",
            "gender": ["Female", "Male"][index % 2],
            "SeniorCitizen": index % 2,
            "Partner": ["No", "Yes"][index % 2],
            "Dependents": ["Yes", "No"][index % 2],
            "tenure": 3 + index,
            "PhoneService": phone,
            "MultipleLines": "No phone service" if phone == "No" else ["No", "Yes"][index % 2],
            "InternetService": internet,
            **service_values,
            "Contract": ["Month-to-month", "One year"][index % 2],
            "PaperlessBilling": ["Yes", "No"][index % 2],
            "PaymentMethod": ["Electronic check", "Mailed check"][index % 2],
            "MonthlyCharges": 30.0 + index,
            "TotalCharges": "" if index == 0 else str((30 + index) * (3 + index)),
            "Churn": "Yes" if churn else "No",
        }
        rows.append(row)
    return pd.DataFrame(rows)


class ModelingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cleaner_converts_target_and_invalid_total_charges(self):
        path = Path(self.temp_dir.name) / "sample.csv"
        sample_rows().to_csv(path, index=False)

        cleaned = load_and_clean_data(path)

        self.assertEqual(len(cleaned), 24)
        self.assertEqual(cleaned[TARGET].tolist()[:4], [0, 1, 0, 1])
        self.assertTrue(pd.isna(cleaned.loc[0, "TotalCharges"]))
        self.assertEqual(set(cleaned[TARGET].unique()), {0, 1})

    def test_cleaner_reports_missing_columns(self):
        path = Path(self.temp_dir.name) / "incomplete.csv"
        pd.DataFrame({"Churn": ["Yes"]}).to_csv(path, index=False)

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            load_and_clean_data(path)

    def test_pipeline_fits_and_predicts_with_raw_features(self):
        data = sample_rows()
        x = data[FEATURES].copy()
        x["TotalCharges"] = pd.to_numeric(x["TotalCharges"], errors="coerce")
        y = data[TARGET].map({"No": 0, "Yes": 1})
        pipeline = make_pipeline(LogisticRegression(max_iter=1000))

        pipeline.fit(x, y)
        probabilities = pipeline.predict_proba(x.iloc[:3])[:, 1]

        self.assertEqual(probabilities.shape, (3,))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        self.assertEqual(set(RAW_NUMERIC_FEATURES + CATEGORICAL_FEATURES), set(FEATURES))

    def test_risk_feature_uses_training_fitted_charge_median(self):
        data = sample_rows()[FEATURES].copy()
        engineer = ChurnFeatureEngineer().fit(data)
        profile = data.iloc[[0]].copy()
        profile.loc[:, "tenure"] = 2
        profile.loc[:, "MonthlyCharges"] = 200.0

        transformed = engineer.transform(profile)

        self.assertEqual(engineer.monthly_charge_median_, 41.5)
        self.assertEqual(int(transformed.iloc[0]["Risk_HighInitialCost"]), 1)
        self.assertIn("AvgMonthlySpend", transformed.columns)

    def test_threshold_is_selected_from_probabilities(self):
        y = pd.Series([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.4, 0.6, 0.9])

        threshold, f1 = best_f1_threshold(y, probabilities)

        self.assertGreaterEqual(threshold, 0.4)
        self.assertLessEqual(threshold, 0.6)
        self.assertAlmostEqual(f1, 1.0)


if __name__ == "__main__":
    unittest.main()

\n