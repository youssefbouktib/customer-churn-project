# Customer Churn Risk Explorer

An end-to-end data science portfolio project that explores customer churn patterns, compares baseline classifiers with a stacking ensemble, evaluates risk scores on a held-out test set, and serves predictions through a Streamlit dashboard.

> **Use responsibly:** this is a learning/demo project built on a sample dataset. A score is a prioritization signal, not certainty or a causal explanation. It should not be used for real customer decisions without validation, monitoring, and review of fairness and business impact.

## Demo

Run the dashboard locally with the included trained model artifact:

```bash
streamlit run app.py
```

The raw CSV is not needed to run the dashboard. It is needed to retrain the model.

### Current held-out test results

These numbers come from the model artifact currently included in this repository. They are for the random 20% test split only; they are not a promise of performance on a new company or future period.

| Model | Decision threshold | Precision | Recall | F1 | ROC AUC | PR AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost + CatBoost + LightGBM stack | 0.305 | 0.536 | 0.773 | 0.633 | 0.846 | 0.664 |

The selected stack reached a training-only out-of-fold F1 of **0.642**, compared with **0.640** for logistic regression and **0.639** for random forest, after selecting each model's threshold on the same training folds. The threshold was selected by maximizing F1. The test set was used only for this final report. Its confusion matrix at that cutoff was 785 true negatives, 250 false positives, 85 false negatives, and 289 true positives.

## What the project does

- Validates and cleans the Telco churn data, including parsing `TotalCharges` and imputing its blank values inside the training pipeline.
- Compares class-weighted logistic regression and random forest baselines with an XGBoost, CatBoost, and LightGBM stacking ensemble.
- Uses logistic regression as the stacking meta-learner, with class imbalance handled by the base learners.
- Uses 3-fold out-of-fold predictions from the training split to select a model and an F1-based decision threshold.
- Keeps a stratified 20% test split out of model and threshold selection, then reports precision, recall, F1, ROC AUC, PR AUC, and the confusion matrix.
- Packages preprocessing and the selected estimator together, so the app uses the same imputation, scaling, and categorical encoding as training.
- Provides descriptive churn charts and a customer-profile prediction form.

## Project structure

```text
customer-churn-project/
├── app.py                         # Streamlit dashboard
├── churn_project/modeling.py      # Data checks, preprocessing, models, metrics
├── scripts/train.py               # Reproducible training and artifact generation
├── tests/test_modeling.py         # Data and inference pipeline checks
├── data/README.md                 # Dataset setup and attribution
├── artifacts/                     # Trained pipeline and evaluation report
├── requirements.txt
└── README.md
```

## Run locally (Windows PowerShell)

From the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The repository includes `artifacts/churn_pipeline.joblib` and `artifacts/model_metrics.json`, so the demo works after installing the dependencies.
The package versions in `requirements.txt` are pinned to the environment used to create and verify the included model artifact.

## Retrain the model

1. Download the file named `WA_Fn-UseC_-Telco-Customer-Churn.csv` from a dataset source. Setup notes and an IBM-hosted source are in [`data/README.md`](data/README.md). Check the source's terms and cite it when using the data.
2. Put the CSV in `data/`.
3. Install dependencies and run:

```powershell
python scripts/train.py
```

The script writes a new model pipeline and its evaluation report under `artifacts/`. It prints held-out test metrics. **Use those generated values in your project description; do not rely on scores from an earlier model run.**

The random split is useful for a reproducible classroom demo, but it does not simulate training on past customers and predicting future customers. Results may change under time-based validation or on a different customer population.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests use small synthetic examples and do not require the raw customer dataset or an extra test framework.

## Deploy

The app is a standard Streamlit application. To publish a hosted demo, push this repository to GitHub with the generated `artifacts/` files, then select `app.py` as the app entry point on a Streamlit-compatible hosting service. Do not commit the raw CSV or your local virtual environment.

## Model and data limitations

- Churn prevalence and patterns reflect this particular sample, not every telecom market or customer base.
- The threshold maximizes F1 on out-of-fold training predictions. A real retention team should select a threshold based on outreach capacity, intervention costs, and the relative cost of missed churners.
- Precision, recall, F1, ROC AUC, and PR AUC answer different questions. Review the confusion matrix and business context instead of relying on accuracy alone.
- Global model factors are associations learned by the model; they do not explain causes and should not be interpreted as an individual-level reason.
- The dataset includes demographic fields. Review fairness and applicable privacy requirements before any real-world use.

## LinkedIn project blurb

> Built an end-to-end customer churn risk explorer using Python, scikit-learn, XGBoost, CatBoost, LightGBM, and Streamlit. I engineered customer behavior features, compared logistic regression and random forest baselines with a stacking ensemble, selected the model and decision threshold using training-only cross-validation, and evaluated the final approach on a held-out test set. The interactive dashboard explores churn patterns and scores sample customer profiles. The project documents its evaluation, assumptions, and limitations.

Before posting, add a screenshot or short screen recording of the dashboard and replace any general language with the actual metrics generated in `artifacts/model_metrics.json`.

\n
